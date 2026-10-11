# Chapter 11 — The numerical-correctness laboratory

> **What you will learn:** how to decide whether two implementations "compute the same thing" when floating
> point never gives exact equality; the metrics the project uses and why; the comparison protocol; the
> tolerance policy; and the two investigations that changed that policy, each started by a failing check and
> ended by finding the real cause.

## 11.1 Three meanings of "the same"

| Notion | Meaning |
|---|---|
| **Mathematical equivalence** | equal in exact real-number arithmetic (proved on paper, e.g. Chapter 8) |
| **Numerical equivalence** | equal within a tolerance justified by the number format and the operations involved |
| **Bitwise equality** | the exact same bits |

Different summation orders give different roundings (Chapter 3), so two correct implementations are generally
**not** bitwise equal. A test pins this: naive and tiled outputs in fp32 differ by under 1e-5, but are not
bitwise equal (`test_numerical_vs_bitwise_equality_are_distinct_notions`). If two mathematically equivalent
implementations are *not* numerically equivalent, something is wrong: a bug, an instability, or a tolerance
that does not fit the situation.

## 11.2 The metrics

> **In the code:** `src/attention_lab/correctness/metrics.py`, function `compare`.

For an output `x` and a reference `r` (both converted to fp64):

* `max_abs`: the largest absolute difference;
* `mean_abs`: the mean absolute difference;
* **`max_abs_normalized` = max|x − r| / max|r|**: the number the tolerances are checked against. Attention
  outputs are weighted averages of values, so `max|r|` is the natural scale;
* `rel_fro = ‖x − r‖ / ‖r‖`: a global relative error that is robust to near-zero entries;
* `max_rel`: elementwise relative error **with an explicit floor**. Entries where `|r| ≤ 10⁻³ · max|r|` are
  excluded, and the excluded fraction is reported. Without the floor, a reference entry of 10⁻¹² makes any
  rounding difference look like a 100 % error;
* `cosine`: cosine similarity of the flattened outputs;
* `nan_count`, `inf_count`;
* `bitwise_equal`.

## 11.3 The protocol for one comparison

> **In the code:** `src/attention_lab/correctness/lab.py`, `evaluate_case`.

1. Draw `q, k, v` in **fp64** from a seeded generator; multiply `q` and `k` by `magnitude`.
2. **Round** them to the test dtype (fp32, bf16, …).
3. The reference is **naive attention in fp64 applied to the already-rounded inputs**. That way the unavoidable
   error of rounding the inputs is not blamed on the algorithm.
4. Run the method in the test dtype, for forward output and for gradients against a fixed random upstream
   gradient.
5. Compute the metrics, look up the tolerance, record pass or fail.

The default matrix (`default_cases`) covers lengths 1, 7, 64, 129 (quick) plus 512 and 1024 (full); head
dimensions 8–128; fp64, fp32 and bf16; causal and non-causal; tile sizes 32 × 48, which deliberately do not
divide the lengths; input magnitudes 0.01 to 30; batch/head shapes `(1,1)` and `(3,5)`; random key padding; and
fully-masked rows. The full run made **795 comparisons** (`python lab.py correctness --full`, run
`20261011-194506_correctness`).

## 11.4 The tolerance policy, and how it got that way

The current policy, against `max_abs_normalized`:

* **fp64:** forward 1e-12, gradients 1e-10.
* **fp32:** forward 2e-5, gradients 2e-4, **or** `4·u·max|S|` (forward) / `16·u·max|S|` (gradients) if larger,
  where `max|S|` is the largest logit and `u = 2⁻²⁴`.
* **bf16 / fp16:** at most **2 × the error of a mixed-precision baseline** plus a small slack (4e-3 forward,
  1.6e-2 gradients for bf16). The baseline is naive attention computed in fp32 on the same rounded inputs, with
  results rounded to bf16.

Neither of the two non-obvious parts was in the first version. Both came from failures that were investigated
rather than "fixed" by loosening.

### Investigation 1: the bf16 baseline that was too lenient

**First version:** the bf16 tolerance was "2 × the error of naive attention *computed in bf16*". The reasoning
was that low-precision methods should be no worse than the obvious low-precision implementation.

**What the run showed:** all methods passed, but the naive-in-bf16 baseline's own error was huge and grew with
input magnitude. The breakdown (L = 96, d = 32):

| input magnitude | naive, all in bf16: fwd error | tiled / SDPA (fp32 accumulators): fwd error |
|---|---|---|
| 1 | ~0.005 | ~0.002 |
| 3 | 0.059 | ~0.002 |
| 10 | **0.147** | ~0.002 |
| 30 (full run) | **0.377** | ~0.003 |

Gradient errors were worse, up to 0.68 at magnitude 10 and 2.24 at magnitude 30.

**Cause:** at magnitude 10 the logits are in the hundreds. bf16 has 8 significant bits, so neighbouring
representable numbers near 256 are 1 apart, and a logit can be off by up to 0.5. The softmax weight
`exp(s − m)` is then off by up to `e^0.5 ≈ 1.65×`. The kernels that keep logits and running sums in fp32 do not
suffer this.

**Why the first policy was wrong:** "2 × a 15 % error" accepts almost anything. A tolerance based on a bad
baseline is meaningless.

**Fix:** use a **mixed-precision baseline** whose error is essentially the unavoidable rounding of the *output*
to bf16 (about 2–3e-3). Under the new policy every fp32-accumulating method passes, and **naive-in-bf16 fails
14 of 53 cases**. That is now reported as a finding, not hidden: *computing attention entirely in bf16 is
inaccurate at realistic logit scales, which is why FlashAttention and the CPU kernel keep these values in fp32.*
A regression test pins it (`test_finding_naive_in_pure_bfloat16_degrades_with_logit_magnitude`).

### Investigation 2: the fp32 "failure" that was the dtype's own floor

**What happened:** after the bf16 fix, one test failed: `sdpa_flash_cpu` in fp32 at input magnitude 10 had
forward error **2.04e-5**, against a tolerance of **2e-5**.

The easy move would have been to change 2e-5 to 3e-5. Instead a hypothesis was formed. A logit `s` in fp32 can
only be stored to about `u·|s|` absolute precision, and that becomes a relative error in its softmax weight of
the same size. If so, the error should scale with the **largest logit**, for *every* implementation, including
the naive formula. Measured (L = 64, d = 32, fp32), error divided by `u·max|S|`:

| input magnitude | max\|S\| | naive | sdpa_math | sdpa_flash_cpu | tiled |
|---|---|---|---|---|---|
| 1 | 4.6 | 1.3 | 1.5 | 1.4 | 1.1 |
| 3 | 41 | 0.6 | 0.7 | 0.6 | 0.6 |
| 10 | 459 | 0.7 | 0.2 | 0.7 | 0.2 |
| 30 | 4135 | 0.1 | 0.2 | 0.1 | 0.1 |

At magnitude 10, **the naive formula in fp32 had exactly the same error, 2.04e-5**, as the fused kernel. The
error tracked the logit scale for every implementation. (It saturates at magnitude 30 because the softmax
becomes nearly one-hot and stops being sensitive.)

**Conclusion:** not a kernel bug, but the fp32 floor for large logits. **Fix:** add the term derived from the
error model, `4·u·max|S|`, which only matters for large logits. The fixed 2e-5 threshold still applies to
ordinary inputs. A regression test pins the evidence
(`test_finding_fp32_error_floor_scales_with_logit_magnitude_for_every_implementation`).

## 11.5 Results of the full run

| method | dtype | cases | pass fwd | pass grad | worst fwd err | median fwd err |
|---|---|---|---|---|---|---|
| sdpa_math / sdpa_flash_cpu / tiled / tiled_recompute | fp64 | 49 each | 100 % | 100 % | ≤ 3.7e-15 | ~3–5e-16 |
| all five exact methods | fp32 | 57 each | 100 % | 100 % | ≤ 6.5e-5 (magnitude 30) | ~2.5–3e-7 |
| the four fp32-accumulating methods | bf16 | 53 each | 100 % | 100 % | ≤ 4.0e-3 | ~2.1e-3 |
| naive (everything in bf16) | bf16 | 53 | **73.6 %** | 94.3 % | **0.377** | 5.7e-3 |

Two more facts from the same run:

* **Unstable softmax demo** (fp32, softmax without subtracting the max): 0 % NaN at max logit 3.4 and 83.8,
  **17.9 % NaN at max logit 335**, 40.3 % at 1340. The stable version's error stayed below 2e-7.
* The exact methods' fp32 output error stayed flat (about 6e-7 to 2e-6) from L = 128 to 2048 in the scaling run;
  it did not grow with length.

## Key points to remember

* Compare with tolerances: correct implementations differ in the last bits.
* The reference is fp64 naive *on the same rounded inputs*.
* The metric is `max|err| / max|ref|`, and relative errors use an explicit floor.
* fp64 1e-12, fp32 2e-5 (or `4·u·max|S|`), bf16 relative to a mixed-precision baseline.
* Investigation 1: a baseline that is itself bad (naive-in-bf16, error up to 38 %) makes tolerances meaningless.
* Investigation 2: when a check fails, test a hypothesis about the cause. Here the fp32 logit-rounding floor was shared by every implementation.
* All fp32-accumulating exact methods passed every check. Naive-in-bf16 failed 14 of 53.
