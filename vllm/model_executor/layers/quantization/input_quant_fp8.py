# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    get_fp8_min_max,
    group_broadcast,
    prep_scale_for_group_broadcast,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)

_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_MIN, _FP8_MAX = get_fp8_min_max()
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


_PADDING_MASK_SCOPE: torch.Tensor | None = None


@contextlib.contextmanager
def padding_mask_scope(is_padding: torch.Tensor):
    """Give the `QuantFP8` modules built in this scope a padding mask.

    `is_padding` is a bool tensor, one entry per padded token row, that the model
    runner refreshes in place every step (True marks a padding row). Dynamic
    per-tensor FP8 scales then ignore those rows: a padded batch keeps whatever an
    earlier step left in its padding rows, so a tensor-wide amax would let that
    stale data set the scale for the real rows.

    The mask is held as a module buffer, so `torch.compile` reads it as a graph
    input and sees the runner's in-place updates; a global tensor would be baked in
    as a constant.
    """
    global _PADDING_MASK_SCOPE
    prev, _PADDING_MASK_SCOPE = _PADDING_MASK_SCOPE, is_padding
    try:
        yield
    finally:
        _PADDING_MASK_SCOPE = prev


# --8<-- [start:quant_fp8]
@CustomOp.register("quant_fp8")
class QuantFP8(CustomOp):
    """Quantize input tensor to FP8 (per-tensor, per-token, per-channel, or per-group).
    This CustomOp supports both static and dynamic quantization.
    """

    # --8<-- [end:quant_fp8]

    def __init__(
        self,
        static: bool,
        group_shape: GroupShape,
        num_token_padding: int | None = None,
        column_major_scales: bool = False,
        tma_aligned_scales: bool = False,
        use_ue8m0: bool | None = None,  # for Torch compile
        compile_native: bool = True,
    ):
        """Args:
        static: static or dynamic quantization
        group_shape: quantization group shape (PER_TOKEN, PER_TENSOR,
            PER_CHANNEL, or arbitrary block size)
        num_token_padding: Pad the token dimension of output to this
            size
        tma_aligned_scales: For group quantization, output scales in
            TMA-aligned layout
        column_major_scales: For group quantization, output scales in
            column major format
        compile_native: Manually compile forward_native if compile mode > None

        """
        super().__init__(compile_native=compile_native)
        self.static = static
        self.group_shape = group_shape
        self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN
        self.num_token_padding = num_token_padding
        self.column_major_scales = column_major_scales
        self.tma_aligned_scales = tma_aligned_scales
        self.use_ue8m0 = is_deep_gemm_e8m0_used() if use_ue8m0 is None else use_ue8m0
        self.use_deep_gemm_supported = is_deep_gemm_supported()

        self.use_aiter = rocm_aiter_ops.is_linear_fp8_enabled()

        # None unless built inside padding_mask_scope; only the dynamic per-tensor
        # scale reads it.
        self.register_buffer(
            "padding_mask",
            _PADDING_MASK_SCOPE if not static else None,
            persistent=False,
        )

        self.is_group_quant = group_shape.is_per_group()
        if self.is_group_quant:
            self.group_size = group_shape.col
        else:
            self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN
            if not static:
                assert group_shape in (GroupShape.PER_TOKEN, GroupShape.PER_TENSOR), (
                    "Only per-token or per-tensor scales are supported for dynamic "
                    "non-group quantization."
                )

    def _dynamic_per_tensor_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Dynamic per-tensor scale over the real rows of `x`."""
        mask = self.padding_mask
        if mask is None or x.dim() != 2 or mask.shape[0] < x.shape[0]:
            x_max = x.abs().max().unsqueeze(-1).to(torch.float32)
        else:
            row_max = torch.linalg.vector_norm(
                x, ord=float("inf"), dim=1, dtype=torch.float32
            )
            x_max = row_max.masked_fill(mask[: x.shape[0]], 0.0).max().unsqueeze(-1)
        return (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.quantization.utils import fp8_utils

        if (
            self.is_group_quant
            and self.use_ue8m0
            and self.use_deep_gemm_supported
            and not self.static
            and self.group_size == 128
            and (DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0)
        ):
            return fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm(
                x,
                group_size=self.group_size,
                use_ue8m0=True,
            )

        if self.is_group_quant and not self.static:
            assert scale is None, "Dynamic group quantization does not use scale"

            return fp8_utils.per_token_group_quant_fp8(
                x,
                group_size=self.group_size,
                column_major_scales=self.column_major_scales,
                tma_aligned_scales=self.tma_aligned_scales,
                dtype=_FP8_DTYPE,
                use_ue8m0=self.use_ue8m0,
            )

        assert (scale is not None) == self.static
        assert scale_ub is None or (
            not self.static
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        if not self.static and self.group_shape == GroupShape.PER_TENSOR:
            scale = self._dynamic_per_tensor_scale(x)
            return ops.scaled_fp8_quant(
                x, scale, num_token_padding=self.num_token_padding
            )

        return ops.scaled_fp8_quant(
            x,
            scale,
            num_token_padding=self.num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            group_shape=(self.group_shape.row, self.group_shape.col)
            if self.static
            else None,
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_aiter_quant = self.use_aiter and scale_ub is None and x.is_contiguous()
        use_aiter_per_tensor_quant = (
            use_aiter_quant and self.group_shape.is_per_tensor()
        )
        use_aiter_per_token_quant = use_aiter_quant and self.group_shape.is_per_token()

        use_aiter_per_group_quant = use_aiter_quant and self.group_shape.is_per_group()

        if use_aiter_per_group_quant:
            return rocm_aiter_ops.group_fp8_quant(x, self.group_size)
        if use_aiter_per_tensor_quant:
            return rocm_aiter_ops.per_tensor_quant(x, _FP8_DTYPE, scale)
        if use_aiter_per_token_quant:
            return rocm_aiter_ops.per_token_quant(x, _FP8_DTYPE, scale)

        # Fall back to the CUDA implementation. Dispatch on the class, not
        # through self: a subclass may override forward_cuda with a different
        # signature (_DecodeConcatQuantFP8 does), and a virtual call would
        # re-enter it with mismatched arguments.
        return QuantFP8.forward_cuda(self, x, scale, scale_ub)

    def forward_xpu(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_group_quant and not self.static:
            from vllm.model_executor.layers.quantization.utils import fp8_utils

            return fp8_utils.per_token_group_quant_fp8(
                x,
                group_size=self.group_size,
                column_major_scales=self.column_major_scales,
                dtype=_FP8_DTYPE,
                use_ue8m0=self.use_ue8m0,
            )
        # Dispatch on the class, for the same reason as in forward_hip.
        return QuantFP8.forward_cuda(self, x, scale, scale_ub, use_triton)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ):
        if self.is_group_quant and not self.static:
            assert scale is None, "Dynamic group quantization does not use scale"
            return self._quantize_group_native(x)

        assert (scale is not None) == self.static
        assert scale_ub is None or (
            not self.static
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        if scale is None:
            if self.group_shape == GroupShape.PER_TOKEN:
                x_max = x.abs().amax(dim=-1)
                x_max = x_max.unsqueeze(-1).to(torch.float32)
                if scale_ub is not None:
                    x_max = x_max.clamp(max=scale_ub)
                scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)
            else:
                scale = self._dynamic_per_tensor_scale(x)
        else:
            scale = prep_scale_for_group_broadcast(scale, x, self.group_shape)

        # Even for dynamic per-token scales,
        # reciprocal performs slightly better than division
        out = (
            x.to(torch.float32)
            * group_broadcast(scale.to(torch.float32), x.shape[-2:]).reciprocal()
        )
        out = out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        # This currently generates an extra Triton kernel in compilation.
        # Fortunately, we don't use padding if compiling.
        # TODO(luka): benchmark torch._scaled_mm to hopefully remove padding
        #  in general.
        if self.num_token_padding is not None:
            padding = max(self.num_token_padding - out.size(0), 0)
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)

        return out, scale

    def _quantize_group_native(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        num_groups = (hidden_dim + self.group_size - 1) // self.group_size
        padded_dim = num_groups * self.group_size

        if padded_dim != hidden_dim:
            padding = padded_dim - hidden_dim
            x = F.pad(x, (0, padding), mode="constant", value=0.0)

        x_grouped = x.view(-1, num_groups, self.group_size)
        absmax = x_grouped.abs().amax(dim=-1, keepdim=True).float()
        scales_raw = (absmax / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)
        scales = (
            torch.exp2(torch.ceil(torch.log2(scales_raw)))
            if self.use_ue8m0
            else scales_raw
        )

        x_scaled = x_grouped / scales
        x_quant = x_scaled.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        x_quant = x_quant.view(-1, padded_dim)
        if padded_dim != hidden_dim:
            x_quant = x_quant[..., :hidden_dim]
        x_quant = x_quant.view(orig_shape)

        scales = scales.squeeze(-1)
        scales = scales.reshape(orig_shape[:-1] + (num_groups,))

        if self.column_major_scales:
            scales = scales.transpose(-2, -1).contiguous().transpose(-1, -2)

        return x_quant, scales
