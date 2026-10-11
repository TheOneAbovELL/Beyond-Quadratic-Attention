"""Synthetic task generators and probes are well-formed."""
import torch

from attention_lab.tasks import accuracy_by_distance, fixed_eval_set, local_copy, make_probe, mqar, run_probe
from attention_lab.tasks.synthetic import KEYS, NOISE, VALUES


def test_local_copy_targets():
    b = local_copy(4, 20, lag=3, generator=torch.Generator().manual_seed(0))
    assert (b.targets[:, :3] == -100).all()
    assert torch.equal(b.targets[:, 3:], b.tokens[:, :-3])


def test_mqar_query_targets_are_bound_values_and_distances_are_correct():
    b = mqar(8, 64, n_pairs=6, n_queries=4, generator=torch.Generator().manual_seed(0))
    for i in range(8):
        tok = b.tokens[i]
        binding = {int(tok[2 * j]): int(tok[2 * j + 1]) for j in range(6)}
        qpos = (b.targets[i] != -100).nonzero().flatten()
        assert len(qpos) == 4
        for p in qpos.tolist():
            key = int(tok[p])
            assert KEYS[0] <= key < KEYS[1]
            assert int(b.targets[i, p]) == binding[key]
            key_pos = [2 * j for j in range(6) if int(tok[2 * j]) == key][0]
            assert int(b.distance[i, p]) == p - key_pos
        assert ((tok[12:] >= NOISE[0]) & (tok[12:] < NOISE[1]) | (b.targets[i, 12:] != -100)).all()


def test_mqar_decoys_are_unbound_keys():
    b = mqar(4, 128, n_pairs=4, n_queries=4, decoy_frac=0.5, generator=torch.Generator().manual_seed(1))
    for i in range(4):
        bound = set(b.tokens[i, 0:8:2].tolist())
        region = b.tokens[i, 8:][b.targets[i, 8:] == -100]
        decoys = region[(region >= KEYS[0]) & (region < KEYS[1])]
        assert len(decoys) > 10
        assert not (set(decoys.tolist()) & bound)


def test_fixed_eval_set_is_deterministic():
    a = fixed_eval_set("mqar", 16, 64, seed=0, n_pairs=8, n_queries=8)
    b = fixed_eval_set("mqar", 16, 64, seed=0, n_pairs=8, n_queries=8)
    assert torch.equal(a.tokens, b.tokens) and torch.equal(a.targets, b.targets)


def test_accuracy_by_distance_buckets():
    b = local_copy(2, 10, lag=3, generator=torch.Generator().manual_seed(0))
    acc = accuracy_by_distance(b.targets.clamp(min=0), b)
    assert acc["accuracy"] == 1.0 and acc["acc_dist_0_8"] == 1.0


def test_probe_construction_target_logit_is_beta():
    q, k, v, t = make_probe(L=32, distance=5, n_trials=3, d=16, beta=7.0)
    s = (q[:, 0, -1] * k[:, 0, t]).sum(-1) / 16 ** 0.5
    torch.testing.assert_close(s, torch.full((3,), 7.0), rtol=1e-5, atol=1e-5)
    assert t == 32 - 1 - 5


def test_probe_exact_attention_retrieves_and_local_window_cannot_see_far():
    near = run_probe("naive", L=64, distance=4, n_trials=16, d=32)
    far_local = run_probe("local_window", L=64, distance=40, n_trials=16, d=32, params={"window": 8})
    assert near["accuracy"] > 0.9
    assert far_local["accuracy"] < 0.2


def test_mqar_v2_values_are_distinct_so_bag_of_values_shortcut_is_one_over_pairs():
    """Regression for generator v1, where repeated values let a binding-free model score ~26 %."""
    from attention_lab.tasks.synthetic import TASK_VERSION, shortcut_baseline

    assert TASK_VERSION >= 2
    b = mqar(64, 64, n_pairs=8, n_queries=8, generator=torch.Generator().manual_seed(3))
    for i in range(64):
        vals = b.tokens[i, 1:16:2]
        assert len(set(vals.tolist())) == 8
    assert abs(shortcut_baseline(b, 8) - 1 / 8) < 1e-9


def test_source_positions_point_at_the_queried_key():
    b = mqar(4, 64, generator=torch.Generator().manual_seed(0))
    for i in range(4):
        for p in (b.targets[i] != -100).nonzero().flatten().tolist():
            assert int(b.tokens[i, int(b.source[i, p])]) == int(b.tokens[i, p])


def test_induction_targets_are_successors_of_earlier_occurrence_with_variable_offset():
    from attention_lab.tasks.synthetic import induction

    b = induction(64, 32, seg=12, min_gap=0, max_gap=8, generator=torch.Generator().manual_seed(0))
    offsets = set()
    for i in range(64):
        for p in (b.targets[i] != -100).nonzero().flatten().tolist():
            s = int(b.source[i, p])
            assert int(b.tokens[i, s]) == int(b.tokens[i, p])          # current token occurred at s
            assert int(b.tokens[i, s + 1]) == int(b.targets[i, p])     # answer = its successor
            assert int(b.distance[i, p]) == p - s
            offsets.add(p - s)
    assert offsets == set(range(12, 21))       # offset varies, so a fixed positional head cannot solve it


def test_induction_distractor_gap_uses_segment_vocabulary():
    from attention_lab.tasks.synthetic import SEG_VOCAB, induction

    b = induction(16, 40, seg=12, min_gap=8, max_gap=8, gap_tokens="segment", generator=torch.Generator().manual_seed(0))
    gap = b.tokens[:, 12:20]
    assert ((gap >= SEG_VOCAB[0]) & (gap < SEG_VOCAB[1])).all()
