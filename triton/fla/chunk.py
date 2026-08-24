import torch

from torch_mlu_ops.triton.fla.cumsum import chunk_local_cumsum
from torch_mlu_ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from torch_mlu_ops.triton.fla.solve_tril import solve_tril
from torch_mlu_ops.triton.fla.wy_fast import recompute_w_u_fwd, prepare_wy_repr_bwd
from torch_mlu_ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_bwd_dhu
from torch_mlu_ops.triton.fla.chunk_o import chunk_fwd_o, chunk_bwd_dqkwg, chunk_bwd_dv_local

def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    allow_tf32: bool = True,
):
    K = k.shape[-1]
    V = v.shape[-1]
    assert K <= 128
    assert V <= 128
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    # [B, T, H] -> [H, B, T]
    # 以提高部分场景的IO效率
    g_trans = g.permute(2,0,1).contiguous()

    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=torch.float32, allow_tf32=allow_tf32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_trans,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        allow_tf32=allow_tf32,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_trans,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        allow_tf32=allow_tf32,
        state_v_first=state_v_first,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_trans,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        allow_tf32=allow_tf32,
        state_v_first=state_v_first,
    )
    return o, final_state, A, g, initial_state

def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    dout: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context = None,
    allow_tf32: bool = True
):
    # [B, T, H] -> [H, B, T]
    # 以提高部分场景的IO效率
    g_trans = g.permute(2,0,1).contiguous()
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g_trans,
        cu_seqlens=cu_seqlens,
        allow_tf32=allow_tf32,
    )

    if cp_context is not None:
        initial_state = expand_h0(initial_state, context=cp_context)

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_trans,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        allow_tf32=allow_tf32,
    )

    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g_trans,
        dout=dout,
        scale=scale,
        cu_seqlens=cu_seqlens,
        allow_tf32=allow_tf32,
    )

    if cp_context is not None:
        # initial_state is None in the CP mode
        # We only need to compute dht of current rank and pass it to the backward kernel
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=q,
            k=k,
            w=w,
            dout=dout,
            dv=dv,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            dht=dht,
            initial_state=initial_state,
            context=cp_context,
        )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g_trans,
        h0=initial_state,
        dht=dht,
        dout=dout,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        allow_tf32=allow_tf32,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g_trans,
        h=h,
        dv=dv,
        dout=dout,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        allow_tf32=allow_tf32,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta.permute(2,0,1).contiguous(),
        g=g_trans,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        allow_tf32=allow_tf32,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0

class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        cp_context = None,
        allow_tf32: bool = True,
        state_v_first: bool = False
    ):
        o, final_state, A, g, initial_state = chunk_gated_delta_rule_fwd(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous() if g is not None else g,
            beta=beta.contiguous() if beta is not None else beta,
            scale=scale,
            initial_state=initial_state.contiguous() if initial_state is not None else initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            allow_tf32=allow_tf32,
        )
        ctx.save_for_backward(q, k, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.cp_context = cp_context
        ctx.allow_tf32 = allow_tf32
        return o.to(q.dtype), final_state

    @staticmethod
    def backward(
        ctx,
        dout: torch.Tensor,
        dht: torch.Tensor,
    ):
        q, k, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous() if g is not None else g,
            beta=beta.contiguous() if beta is not None else beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state.contiguous() if initial_state is not None else initial_state,
            dout=dout.contiguous(),
            dht=dht.contiguous(),
            cu_seqlens=cu_seqlens,
            cp_context=ctx.cp_context,
            allow_tf32=ctx.allow_tf32,
        )
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None, None, None, None
