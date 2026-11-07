"""A faster engine for the registered reference runtime (detrt), bit-identical to it.

detrt evaluates every fp32 + - * / sqrt as a float64 op rounded once to float32, so that no
platform's fast-math can change a bit. On a GPU that costs most of a job's time: a 3080 runs
float64 at 1/64 of its float32 rate, and each op also pays two dtype-conversion kernels
(bench/lambda_det.py: 40% of kernel time in conversions, 26% in float64 math, 3% in GEMMs).

IEEE-754 float32 + - * / sqrt are correctly rounded, which is exactly what detrt's emulation
computes (one rounding of the exact result; float64 then float32 double rounding is
innocuous for these ops since 53 >= 2*24 + 2). So where the hardware and kernels are IEEE
(no fast-math, no flush-to-zero, no fused multiply-add across ops), native float32 gives the
same bits. tests/test_fp32_native.py checks this per op on each GPU; tests/test_detgpu.py
checks whole stages against detrt.fast. This module is not part of the runtime hash: the
chain's referee and detrt stay the definition, and this is only allowed to match them.

Stage outputs are also kept on the GPU between layer groups (no CPU round trip per group).
"""
