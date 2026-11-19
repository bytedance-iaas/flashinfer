"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_experimental_api

# Thin experimental entry points; validation of the backend contract, the
# workspace policy, launch binding and JIT registration live in
# flashinfer.experimental.cake_dsa_indexer (backend="cake") and
# flashinfer.experimental.sm120_dsa_indexer (backend="sm120").

_FEATURE = (
    "DSA indexer top-k selection (cake: BF16 32x128 on SM100/SM103/SM107; "
    "sm120: FP8 / MXFP4 32/64x128 on SM120)"
)


@flashinfer_experimental_api(feature=_FEATURE)
def dsa_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    *,
    top_k: int = 2048,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    return_scores: bool = True,
    backend: str = "cake",
    k_scale: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    context_lens: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
    diagnostic_counts: Optional[torch.Tensor] = None,
    recovery_splits: Optional[str] = None,
    sample_target: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Fused DSA indexer scoring and deterministic exact top-k selection
    (DeepSeek Sparse Attention indexer, training), SM100 / SM103 / SM107 only.

    For query ``t`` and key ``j`` of the same packed segment

    ``s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] * k[j, d])``

    with FP32 products, accumulation and score arithmetic (no BF16 intermediate
    rounding; the reduction scheme is fixed and documented in
    ``flashinfer/experimental/cake_dsa_indexer/README.md``).  Every row selects
    the exact global top ``min(top_k, visible)`` keys of its visible prefix,
    ranked by score descending then key id descending (``+0.0 == -0.0`` for
    ranking), and returns them in ascending id order with aligned scores.

    Parameters
    ----------
    q : torch.Tensor
        BF16 ``[T, 32, 128]`` indexer queries after RoPE (contiguous).
    k : torch.Tensor
        BF16 ``[Tkv, 128]`` packed indexer keys after normalization and RoPE.  A
        row-strided view (for example ``packed[:, :128]`` of a ``[Tkv, 704]``
        tensor) is read in place; the row stride must be a multiple of 8
        elements.
    w : torch.Tensor
        FP32 ``[T, 32]`` signed head weights, already scaled by ``32 ** -0.5``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        int32 ``[S + 1]`` independent query and key segment boundaries on the
        device (``cu_seqlens_q[-1] == T``, ``cu_seqlens_k[-1] == Tkv``).  They
        are not read on the host.
    top_k : int
        Output slots per query, ``1 <= top_k <= 4096`` (default 2048).
    softmax_scale : Optional[float]
        Positive scale applied to the per-head logits; defaults to ``128 ** -0.5``.
    q_causal_offsets : Optional[torch.Tensor]
        int64 ``[S]`` per-segment query offsets on the device.  Key ``j`` is
        visible to the query at segment-local position ``u`` iff ``j < Lk`` and
        ``j < floor((offset + u + 1) / ratio)``; without offsets ``offset = Lk -
        Lq`` for ``ratio == 1`` (queries are the tail of their key prefix) and
        ``0`` for compressed keys.  Negative offsets, empty segments and rows
        without visible keys are defined (the row is all padding).
    ratio : int
        Key compression ratio, ``>= 1`` (normally 1).
    max_seqlen_q, max_seqlen_k : Optional[int]
        Host mirrors of the longest segments; accepted for signature parity and
        not read by any launch decision.
    workspace_buffer : Optional[torch.Tensor]
        Scratch of at least :func:`dsa_indexer_topk_workspace_size` bytes
        (uint8, 8-byte aligned); allocated through the caching allocator when
        omitted.  The bound is independent of ``T``, ``Tkv`` and ``S``.
    indices, scores : Optional[torch.Tensor]
        Caller-owned ``[T, top_k]`` int32 / float32 outputs; allocated when
        omitted.
    return_scores : bool
        Return selected FP32 scores with indices.  ``False`` skips score writes
        for the SM120 backend.
    backend : str
        ``"cake"`` for the BF16 training form or ``"sm120"`` for the FP8 /
        MXFP4 serving form on SM12x GPUs.
    k_scale, q_scale : Optional[torch.Tensor]
        Key scales and optional MXFP4 query scales for ``backend="sm120"``.
    block_table, context_lens : Optional[torch.Tensor]
        Supplying both selects the paged SM120 form; omitting both selects the
        ragged form.
    enable_pdl : Optional[bool]
        Programmatic dependent launch for ``backend="sm120"``.
    min_seqlen_k : Optional[int]
        Certified host lower bound on visible keys of every nonempty query
        segment, for ragged SM120 calls only. The caller must derive it from
        exact metadata for this call; a bound above 48K omits resident scratch.
        It never replaces ``q_causal_offsets`` or changes causal positions.
    diagnostic_counts : Optional[torch.Tensor]
        Ragged SM120-only caller-owned contiguous int32 CUDA output of shape
        ``[ceil(T / 4096), 8]``. Records per scan chunk: recovery rows, exact
        recovery failures, insufficient candidates, candidate overflow, scanned
        rows, rows with sampled cutoff ties, empty recovery launches, and the
        number of sampled keys tied at the cutoff. Resident-only chunks are
        zero. No host synchronization is added; read after the call completes.
        Omit for normal execution. Concurrent calls need distinct outputs.
    recovery_splits : Optional[str]
        Ragged SM120 recovery scheduling: ``None`` / ``"adaptive"`` derives
        key splits on the device from the actual flagged-row count; ``"legacy"``
        retains the host estimate of one recovery row per 256 query rows.
        Both paths compute the same native scores and deterministic exact TopK.
    sample_target : Optional[int]
        Ragged SM120-only threshold candidate target, between ``top_k`` and
        6144. ``None`` retains the existing target. Exact scoring, candidate
        capacity and full recovery are unchanged; this controls sampling work.

    Returns
    -------
    indices : torch.Tensor
        int32 ``[T, top_k]`` segment-local key ids, ascending within the row,
        unique; unused tail slots hold ``-1``.
    scores : torch.Tensor
        FP32 ``[T, top_k]`` scores aligned with ``indices`` (computed FP32
        bits; a selected ``-0.0`` is not normalized); tail slots hold ``-inf``.

    Identical inputs give identical ids and score bits; the result does not
    depend on the internal query / key partition.  No ``[T, Tkv]`` matrix is
    materialized and the call performs no host synchronization, so it can be
    captured into a CUDA graph.  Finite inputs are the normal domain: with
    NaN / inf / overflowing inputs the computed bits are returned, NaN scores
    rank below every other score and the kernels never hang.
    """
    paged = block_table is not None or context_lens is not None
    if diagnostic_counts is not None and (backend != "sm120" or paged):
        raise ValueError("diagnostic_counts requires ragged backend='sm120'")
    if recovery_splits is not None and (backend != "sm120" or paged):
        raise ValueError("recovery_splits requires ragged backend='sm120'")
    if sample_target is not None and (backend != "sm120" or paged):
        raise ValueError("sample_target requires ragged backend='sm120'")
    if paged and backend != "sm120":
        raise ValueError(
            "the paged form (block_table, context_lens) needs backend='sm120'"
        )
    if backend == "sm120" and softmax_scale is not None:
        raise ValueError(
            "backend='sm120' takes all scales folded into w; pass softmax_scale=None"
        )
    if backend == "sm120" and paged:
        if block_table is None or context_lens is None:
            raise ValueError("the paged form needs both block_table and context_lens")
        if cu_seqlens_q is not None or cu_seqlens_k is not None or k_scale is not None:
            raise ValueError(
                "the paged form takes cu_seqlens_q=None, cu_seqlens_k=None and "
                "k_scale=None (the scales live in the pages)"
            )
        if (
            q_causal_offsets is not None
            or ratio != 1
            or max_seqlen_k is not None
            or min_seqlen_k is not None
        ):
            raise ValueError(
                "q_causal_offsets, ratio and max_seqlen_k apply to the ragged form only"
            )
        from .experimental.sm120_dsa_indexer import dsa_indexer_det_topk_paged

        return dsa_indexer_det_topk_paged(
            q,
            k,
            w,
            context_lens,
            block_table,
            top_k=top_k,
            max_context_len=block_table.shape[1] * k.shape[1],
            page_rows=k.shape[1],
            q_scale=q_scale,
            indices=indices,
            scores=scores,
            return_scores=return_scores,
            enable_pdl=enable_pdl,
        )
    if backend == "sm120":
        if k_scale is None:
            raise ValueError("backend='sm120' requires k_scale")
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("the ragged form needs cu_seqlens_q and cu_seqlens_k")
        from .experimental.sm120_dsa_indexer import (
            dsa_indexer_det_topk as sm120_dsa_indexer_topk,
        )

        return sm120_dsa_indexer_topk(
            q,
            k,
            w,
            cu_seqlens_q,
            cu_seqlens_k,
            k_scale=k_scale,
            q_scale=q_scale,
            top_k=top_k,
            q_causal_offsets=q_causal_offsets,
            ratio=ratio,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_k=min_seqlen_k,
            diagnostic_counts=diagnostic_counts,
            recovery_splits=recovery_splits,
            sample_target=sample_target,
            indices=indices,
            scores=scores,
            return_scores=return_scores,
            enable_pdl=enable_pdl,
        )
    if min_seqlen_k is not None:
        raise ValueError("min_seqlen_k is only used by ragged backend='sm120'")
    if k_scale is not None or q_scale is not None:
        raise ValueError("k_scale and q_scale are only used by backend='sm120'")
    if backend == "cake":
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("backend='cake' needs cu_seqlens_q and cu_seqlens_k")
        if not return_scores and scores is not None:
            raise ValueError("scores given with return_scores=False")
        from .experimental.cake_dsa_indexer.cake_backend import (
            dsa_indexer_topk as cake_dsa_indexer_topk,
        )

        indices, cake_scores = cake_dsa_indexer_topk(
            q,
            k,
            w,
            cu_seqlens_q,
            cu_seqlens_k,
            top_k=top_k,
            softmax_scale=softmax_scale,
            q_causal_offsets=q_causal_offsets,
            ratio=ratio,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            workspace_buffer=workspace_buffer,
            indices=indices,
            scores=scores,
        )
        return indices, cake_scores if return_scores else None
    raise ValueError("DSA indexer top-k supports backend='cake' or 'sm120'")


# The upstream SM120 backend introduced this spelling after v0.7.2.post1.
dsa_indexer_det_topk = dsa_indexer_topk


def dsa_indexer_topk_workspace_size(
    top_k: int = 2048, device: Optional[torch.device] = None
) -> int:
    r"""Explicit workspace bound of :func:`dsa_indexer_topk` in bytes.

    ``grid x queries_per_cta x candidate_capacity(top_k) x entry_bytes`` with
    ``grid`` = the SM count of ``device`` (the persistent grid) and the
    candidate-buffer policy of the program registered for the device;
    independent of ``T``, ``Tkv`` and the segment count (for example about
    37 MiB at ``top_k = 2048`` on a 148-SM device).  Raises
    ``NotImplementedError`` while no generated program is registered for the
    device's architecture.
    """
    from .experimental.cake_dsa_indexer.cake_backend import (
        dsa_indexer_workspace_size as cake_dsa_indexer_workspace_size,
    )

    return cake_dsa_indexer_workspace_size(top_k, device)
