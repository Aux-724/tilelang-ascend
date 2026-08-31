import pytest
import torch

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def scalar_sqrt_rsqrt(length=8, dtype="float32"):
    @T.prim_func
    def main(
        input_tensor: T.Tensor((length,), dtype),
        output_tensor: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            input_ub = T.alloc_ub((length,), dtype)
            output_ub = T.alloc_ub((length,), dtype)
            T.copy(input_tensor, input_ub)
            for i in T.serial(length):
                value = input_ub[i]
                output_ub[i] = T.sqrt(value) + T.rsqrt(value + 1.0)
            T.copy(output_ub, output_tensor)

    return main


def test_scalar_sqrt_rsqrt_codegen():
    compiled = tilelang.compile(
        scalar_sqrt_rsqrt(),
        out_idx=[1],
        pass_configs=PASS_CONFIGS,
        target="ascendc",
    )
    source = compiled.get_kernel_source()
    assert "sqrt(" in source
    assert "1.0f /" in source
    assert "== 0.0f ? 0.0f" in source


def test_scalar_sqrt_rsqrt_runtime():
    compiled = tilelang.compile(
        scalar_sqrt_rsqrt(dtype="float32"),
        out_idx=[1],
        pass_configs=PASS_CONFIGS,
        target="ascendc",
    )
    input_tensor = torch.arange(8, dtype=torch.float32, device="npu")
    output_tensor = compiled(input_tensor)
    expected = torch.sqrt(input_tensor) + torch.rsqrt(input_tensor + 1.0)
    torch.testing.assert_close(output_tensor, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
