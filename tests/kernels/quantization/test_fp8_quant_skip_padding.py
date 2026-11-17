# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic per-tensor FP8 quantization must not let padding rows set the scale.

The model runner pads a batch to a CUDA graph size, and padding rows hold
whatever an earlier step left in them. A QuantFP8 built inside
`padding_mask_scope` reads that mask so that only real tokens contribute to the
scale.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8,
    padding_mask_scope,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="dynamic FP8 quantization needs CUDA"
)

HIDDEN_SIZE = 256
STALE_VALUE = 1000.0
FORWARDS = ["forward_native", "forward_cuda"]


def _padded_batch(num_real: int, num_padding: int, dtype: torch.dtype):
    real = torch.randn(num_real, HIDDEN_SIZE, device="cuda", dtype=dtype)
    stale = torch.full(
        (num_padding, HIDDEN_SIZE), STALE_VALUE, device="cuda", dtype=dtype
    )
    is_padding = torch.zeros(num_real + num_padding, dtype=torch.bool, device="cuda")
    is_padding[num_real:] = True
    return real, torch.cat([real, stale]), is_padding


def _quant(is_padding: torch.Tensor | None = None) -> QuantFP8:
    if is_padding is None:
        return QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)
    with padding_mask_scope(is_padding):
        return QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)


def _as_bytes(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.uint8)


@pytest.mark.parametrize("forward", FORWARDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_real,num_padding", [(5, 3), (1, 7), (16, 16)])
@torch.inference_mode()
def test_padding_rows_do_not_set_the_scale(
    default_vllm_config, forward, dtype, num_real, num_padding
):
    set_random_seed(0)
    real, padded, is_padding = _padded_batch(num_real, num_padding, dtype)

    expected_q, expected_scale = getattr(_quant(), forward)(real)
    q, scale = getattr(_quant(is_padding), forward)(padded)

    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    assert torch.equal(_as_bytes(q[:num_real]), _as_bytes(expected_q))


@pytest.mark.parametrize("forward", FORWARDS)
@torch.inference_mode()
def test_all_real_rows_match_the_unmasked_result(default_vllm_config, forward):
    """A mask that marks every row as real must not change the result."""
    set_random_seed(0)
    x = torch.randn(32, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    is_padding = torch.zeros(32, dtype=torch.bool, device="cuda")

    expected_q, expected_scale = getattr(_quant(), forward)(x)
    q, scale = getattr(_quant(is_padding), forward)(x)

    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(scale, ops.scaled_fp8_quant(x)[1], rtol=0, atol=0)
    assert torch.equal(_as_bytes(q), _as_bytes(expected_q))


@pytest.mark.parametrize("forward", FORWARDS)
@torch.inference_mode()
def test_built_outside_the_scope_keeps_the_whole_tensor_scale(
    default_vllm_config, forward
):
    """A layer built outside the scope, such as a draft model's, ignores the mask."""
    set_random_seed(0)
    _, padded, is_padding = _padded_batch(5, 3, torch.bfloat16)
    with padding_mask_scope(is_padding):
        pass
    quant = _quant()

    _, expected_scale = ops.scaled_fp8_quant(padded)
    _, scale = getattr(quant, forward)(padded)

    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("forward", FORWARDS)
@torch.inference_mode()
def test_mask_shorter_than_the_batch_is_ignored(default_vllm_config, forward):
    set_random_seed(0)
    _, padded, _ = _padded_batch(5, 3, torch.bfloat16)
    short_mask = torch.zeros(padded.shape[0] - 2, dtype=torch.bool, device="cuda")
    short_mask[-1] = True

    _, expected_scale = ops.scaled_fp8_quant(padded)
    _, scale = getattr(_quant(short_mask), forward)(padded)

    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("compiled", [False, True])
@torch.inference_mode()
def test_in_place_mask_updates_are_seen(default_vllm_config, compiled):
    """The runner rewrites the mask in place each step. Compiled code must read the
    current contents, not the contents at trace time."""
    set_random_seed(0)
    real, padded, is_padding = _padded_batch(5, 3, torch.bfloat16)
    quant = _quant(is_padding)
    forward = quant.forward_native
    if compiled:
        forward = torch.compile(forward, fullgraph=True, dynamic=False)

    _, whole_scale = ops.scaled_fp8_quant(padded)
    _, real_scale = ops.scaled_fp8_quant(real)

    _, scale = forward(padded)
    torch.testing.assert_close(scale, real_scale, rtol=0, atol=0)

    is_padding.zero_()  # every row is now real
    _, scale = forward(padded)
    torch.testing.assert_close(scale, whole_scale, rtol=0, atol=0)

    is_padding[5:] = True  # back to 5 real rows
    _, scale = forward(padded)
    torch.testing.assert_close(scale, real_scale, rtol=0, atol=0)
