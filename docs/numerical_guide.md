# Numerical guide

## 1. Three notions of "the same"

| Notion | Meaning | Example here |
|---|---|---|
| **Mathematical equivalence** | equal in exact real arithmetic | tiled vs naive attention (proof in `online_softmax.md`) |
| **Numerical equivalence** | equal within a tolerance justified by the dtype and the operation | tiled vs naive: max normalized error ≈ 1e-6 in fp32 |
| **Bitwise equality** | identical bit patterns | **not** expected across different reduction orders; tested to be false (`test_numerical_vs_bitwise_equality_are_distinct_notions`) |

Mathematical equivalence plus numerical *non*-equivalence means there is a bug, a
stability problem, or an unjustified tolerance. Investigate before loosening anything.

## 2. Sources of floating-point difference in attention

1. **Reduction order.** `Σ_j e^{s_j}` and `Σ_j P_j v_j` summed in a different order
   (per tile, per thread, or by a blocked matmul) round differently. The error per
   reduction of `n` terms is bounded by about `n·u·Σ|terms|` (worst case). Typical
   errors grow more like `√n·u`.
2. **Extra operations.** Online softmax rescales the accumulator once per tile
   (`α·acc`), which adds one rounding each time.
3. **Logit precision.** An absolute logit error `δ` becomes a relative probability
   error `≈ δ`, because `e^{s+δ} = e^s(1+δ+…)`. Logits are *large* numbers whose
   *differences* matter. Storing them in a format with few mantissa bits loses
   absolute precision as their magnitude grows.
4. **Matmul kernels.** MKL/oneDNN may use different blocking or FMA patterns
   depending on shape and thread count. Results can even differ run to run when
   thread scheduling changes the reduction split.

Unit roundoff `u`: fp64 1.1e-16, fp32 6.0e-8, fp16 4.9e-4, bf16 3.9e-3.

## 3. Comparison protocol (`correctness/lab.py`)

* The reference is naive attention in **float64**, applied to the *already rounded*
  low-precision inputs. Input quantization therefore does not count as algorithm error.
* Primary metric: `max_abs_normalized = max|x − r| / max|r|`. Attention outputs
  are convex combinations of values, so `max|r|` is the natural scale.
* Also reported: mean absolute error, `rel_fro = ‖x−r‖_F / ‖r‖_F`, cosine
  similarity, NaN/Inf counts, and an elementwise relative error with an explicit floor.
  Entries with `|r| ≤ 1e-3·max|r|` are excluded, and the excluded fraction is
  reported. Without the floor, any rounding of a near-zero entry looks like a
  100 % error.
* Gradients: the worst of `dq, dk, dv` under the same metric, for a fixed random
  upstream gradient.

## 4. Tolerance policy

* **fp64**: forward 1e-12, gradients 1e-10.
* **fp32**: forward 2e-5, gradients 2e-4. That is a few hundred `u`, which covers
  reductions over `L ≤ 8192` and `d ≤ 256` with typical error growth. Gradients
  chain more operations, hence the extra 10×.
* **Logit term (fp32/fp64)**: the threshold is raised to `4·u·max|S|` (forward) and
  `16·u·max|S|` (gradients) when that is larger. A logit `s` is representable only to
  `≈ u·|s|` absolute precision, and `exp` turns that into a relative probability error
  of the same size, *in any implementation*.

  **Investigation that led here.** `sdpa_flash_cpu` first failed the fixed fp32
  threshold at input magnitude 10, with error 2.04e-5 against a 2e-5 tolerance.
  Measured fp32 forward errors (L=64, d=32), as a multiple of `u·max|S|`:

  | input magnitude | max\|S\| | naive | sdpa_math | sdpa_flash_cpu | tiled |
  |---|---|---|---|---|---|
  | 1 | 4.6 | 1.3 | 1.5 | 1.4 | 1.1 |
  | 3 | 41 | 0.6 | 0.7 | 0.6 | 0.6 |
  | 10 | 459 | 0.7 (2.04e-5) | 0.2 | 0.7 (2.04e-5) | 0.2 |
  | 30 | 4135 | 0.1 | 0.2 | 0.1 | 0.1 |

  The naive formula evaluated in fp32 has *exactly* the fused kernel's error at
  magnitude 10. The error tracks the logit scale for every implementation, which
  identifies it as a property of the dtype, not a defect of the kernel. (It
  saturates at magnitude 30 because the softmax becomes nearly one-hot.) The
  tolerance was therefore extended with a term derived from this error model,
  rather than raised globally. For ordinary inputs (max|S| ≲ 100) the fixed
  threshold still applies. Test:
  `test_finding_fp32_error_floor_scales_with_logit_magnitude_for_every_implementation`.
* **bf16 / fp16**: a fixed threshold is a poor model. We compare against a
  **mixed-precision baseline**: naive attention computed in fp32 on the rounded
  inputs, with results rounded to bf16. Its error is essentially the unavoidable
  output rounding (~2–3e-3). An implementation passes if its error is
  `≤ 2 × baseline + slack` (slack 4e-3 forward, 1.6e-2 gradients for bf16).

### How this policy was arrived at (an investigation, not a loosening)

The first version used "naive attention computed *in bf16*" as the low-precision
baseline. The correctness run showed that baseline's forward error grows with input
magnitude: about **5e-3 at magnitude 1, 0.059 at magnitude 3, and 0.147 at magnitude 10**
(gradient error up to **0.68**). The tiled and SDPA kernels stayed flat at about
**2e-3**. Cause: at magnitude 10, logits reach the hundreds, where adjacent bf16 values
are ~1.0 apart, so `exp(s−m)` can be off by a factor `e^{0.5}`. A tolerance of
"2× a 15 % error" would have accepted almost anything. So the baseline was
replaced by the mixed-precision one, and the naive-in-bf16 result is now reported
as a **finding**: computing attention in bf16 without fp32 logits and statistics is
inaccurate at realistic logit scales. That is exactly why FlashAttention and the ATen
CPU kernel keep `S`, `m` and `l` in fp32. The tiled implementation here does the same
(`_acc_dtype`). Test:
`test_finding_naive_in_pure_bfloat16_degrades_with_logit_magnitude`.

## 5. Stability features in the code

| Hazard | Protection | Where |
|---|---|---|
| `exp` overflow for large logits | subtract the row max (or running max) | `safe_row_max`, `online_softmax_update` |
| `−∞ − (−∞) = NaN` for fully-masked rows or tiles | shift by 0 when max is `−∞` | same |
| `0/0` for fully-masked rows | divide by 1 when `l = 0`; output 0 by policy | all exact methods |
| bf16 logit precision | fp32 accumulators in tiled code | `_acc_dtype` |
| tiny denominators in linear attention | `+ eps` (1e-6); `φ > 0` guarantees positivity | `linear.py` |
| gradient noise through the softmax shift | the max is detached (output invariant to it) | `safe_row_max` |

`unstable_softmax_demo` measures what happens without max-subtraction: in fp32,
NaNs appear once logits exceed ~88 (see the correctness report).

## 6. dtype behaviour on this machine

* fp64: used for references; every exact method agrees with naive to ~1e-15.
* fp32: the default comparison dtype; every exact method agrees to ≲ 2e-5
  normalized, including magnitudes up to 30.
* bf16: supported by every implementation. Accuracy depends on accumulator precision
  (§4). Speed is a separate, measured question: this CPU has no native bf16
  arithmetic (`shapes_dtypes.yaml`).
* fp16: tolerance policy defined, but not included in the default matrix. CPU
  fp16 kernels are not a target of this study.
