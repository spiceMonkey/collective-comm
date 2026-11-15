"""Regression tests for ``core.collective_cost`` (stdlib ``unittest`` only).

Run from the repo root:

    .venv/bin/python -m unittest discover -s code/tests -v

The numeric oracles are the canonical anchors and worked examples in
``documentation/modeling/``; if one fails, re-verify against the doc before
touching the expected value.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import collective_cost as cc  # noqa: E402
from util import Anchors, ETA_PROFILES, run_all  # noqa: E402

M, G, ALPHA, BW = 16e6, 512, 0.5e-6, 900e9
DIMS = (8, 8, 8)


def us(t: float) -> float:
    return t * 1e6


class CanonicalAnchors(unittest.TestCase):
    """04_in_network_collectives.md §3.1 and 05_contention_and_congestion.md §5.1."""

    def test_all_anchors_within_tolerance(self):
        for label, value, target in cc.canonical_anchors():
            with self.subTest(label=label):
                self.assertAlmostEqual(value, target, delta=cc.ANCHOR_TOLERANCE_US)

    def test_exact_values(self):
        self.assertAlmostEqual(us(cc.ring_all_reduce(M, G, ALPHA, BW)), 546.486, places=2)
        self.assertAlmostEqual(us(cc.tree_all_reduce(M, G, ALPHA, BW, pipelined=True)), 44.556, places=2)
        self.assertAlmostEqual(us(cc.inc_all_reduce(M, ALPHA, BW)), 18.778, places=2)
        self.assertAlmostEqual(us(cc.torus_all_reduce(M, DIMS, ALPHA, BW)), 56.486, places=2)


class DoubleBinaryTree(unittest.TestCase):
    """01_collective_algorithms.md §5.2: single-port n_β = 2, multi-port halves it."""

    def test_port_models(self):
        log_g = 9
        base = M / BW
        self.assertAlmostEqual(cc.tree_all_reduce(M, G, ALPHA, BW),
                               2 * log_g * ALPHA + 2 * log_g * base)
        self.assertAlmostEqual(cc.tree_all_reduce(M, G, ALPHA, BW, pipelined=True),
                               2 * log_g * ALPHA + 2 * base)
        self.assertAlmostEqual(cc.tree_all_reduce(M, G, ALPHA, BW, multi_port=True),
                               2 * log_g * ALPHA + log_g * base)
        self.assertAlmostEqual(cc.tree_all_reduce(M, G, ALPHA, BW, pipelined=True, multi_port=True),
                               2 * log_g * ALPHA + base)

    def test_inc_halves_software_bw_term(self):
        big = 1e12  # BW-bound regime
        ratio = cc.tree_all_reduce(big, G, ALPHA, BW, pipelined=True) / cc.inc_all_reduce(big, ALPHA, BW)
        self.assertAlmostEqual(ratio, 2.0, places=3)

    def test_ring_crossover(self):
        # M* = N (N - 1 - ceil(log2 N)) α BW  (01 §5.3): ~2.07 GB at N=72, ~116 GB at N=512
        for n, log_n, expect_gb in ((72, 7, 2.07), (512, 9, 115.7)):
            m_star = n * (n - 1 - log_n) * ALPHA * BW
            self.assertAlmostEqual(m_star / 1e9, expect_gb, delta=0.05)
            self.assertAlmostEqual(cc.ring_all_reduce(m_star, n, ALPHA, BW),
                                   cc.tree_all_reduce(m_star, n, ALPHA, BW, pipelined=True))


class AllToAll(unittest.TestCase):
    def test_ring_relay_is_one_dim_torus(self):
        for n in (2, 3, 8, 64, 512):
            self.assertAlmostEqual(cc.ring_relay_a2a(M, n, ALPHA, BW),
                                   cc.torus_a2a(M, (n,), ALPHA, BW))

    def test_ring_relay_bisection_load(self):
        # Brute-force busiest directed edge, antipodal chunks split both ways.
        def busiest(n):
            load = [0.0] * (2 * n)  # [right edges | left edges]
            for s in range(n):
                for d in range(n):
                    r = (d - s) % n
                    if r == 0:
                        continue
                    if 2 * r < n:
                        ways = [(1.0, +1, r)]
                    elif 2 * r > n:
                        ways = [(1.0, -1, n - r)]
                    else:
                        ways = [(0.5, +1, r), (0.5, -1, r)]
                    for w, sign, hops in ways:
                        for h in range(hops):
                            node = (s + sign * h) % n
                            load[node if sign > 0 else n + node] += w
            return max(load)
        for n in (4, 8, 16, 64):
            self.assertAlmostEqual(busiest(n), n * n / 8)  # chunks of size M/n → (n/8)·M
        # G = 2: a single link carries M/2 each way.
        self.assertAlmostEqual(cc.ring_relay_a2a(M, 2, 0.0, BW), 0.5 * M / BW)

    def test_torus_coefficients(self):
        base = M / BW
        self.assertAlmostEqual(cc.torus_a2a(M, (8, 8, 8), 0.0, BW), 1.0 * base)
        self.assertAlmostEqual(cc.torus_a2a(M, (16, 16, 16), 0.0, BW), 2.0 * base)
        self.assertAlmostEqual(cc.torus_a2a(M, (4, 4, 4), 0.0, BW), 0.5 * base)
        self.assertAlmostEqual(cc.torus_a2a(M, (8, 8, 8), 0.0, BW, wraparound=False), 2.0 * base)
        # d = 2: wraparound link coincides with the direct link → mesh cut.
        self.assertAlmostEqual(cc.torus_a2a(M, (2, 2, 2), 0.0, BW), 0.5 * base)
        self.assertEqual(cc.torus_a2a(M, (2, 2, 2), ALPHA, BW),
                         cc.torus_a2a(M, (2, 2, 2), ALPHA, BW, wraparound=False))
        # 3×2: the size-2 dim is the worst cut (1/2 > 3/8).
        self.assertAlmostEqual(cc.torus_a2a(M, (3, 2), 0.0, BW), 0.5 * base)
        self.assertAlmostEqual(cc.torus_a2a(M, (8, 2), 0.0, BW), 1.0 * base)

    def test_inc_alpha_is_two_switch_hops(self):
        base = M / BW
        for fn in (cc.inc_broadcast, cc.inc_reduce, cc.inc_all_reduce):
            self.assertAlmostEqual(fn(M, ALPHA, BW), 2 * ALPHA + base)
        for fn in (cc.inc_all_gather, cc.inc_reduce_scatter, cc.inc_a2a):
            self.assertAlmostEqual(fn(M, G, ALPHA, BW), 2 * ALPHA + 511 / 512 * base)

    def test_alpha_terms(self):
        self.assertAlmostEqual(cc.torus_a2a(0.0, DIMS, ALPHA, BW), 12 * ALPHA)
        self.assertAlmostEqual(cc.torus_a2a(0.0, DIMS, ALPHA, BW, wraparound=False), 21 * ALPHA)
        self.assertAlmostEqual(cc.ring_relay_a2a(0.0, G, ALPHA, BW), 256 * ALPHA)
        self.assertAlmostEqual(cc.pairwise_a2a(0.0, G, ALPHA, BW), 511 * ALPHA)
        self.assertAlmostEqual(cc.bruck_a2a(M, G, ALPHA, BW), 9 * ALPHA + 4.5 * M / BW)

    def test_inc_a2a(self):
        self.assertAlmostEqual(us(cc.inc_a2a(M, G, ALPHA, BW)), 1.0 + 511 / 512 * 17.7778, places=2)


class Hierarchical(unittest.TestCase):
    """03_hierarchical_topologies.md §2.1 case study: N=144, L=2, NVL72 + IB."""

    ARGS = dict(M=16e6, N=144, L=2, alpha_inner=0.5e-6, bw_inner=900e9,
                alpha_outer=2e-6, bw_outer=100e9)

    def test_ar_case_study(self):
        self.assertAlmostEqual(us(cc.hierarchical_all_reduce_ring_ring(**self.ARGS)), 112.28, delta=0.01)

    def test_ag_and_rs_are_half_of_ar(self):
        ar = cc.hierarchical_all_reduce_ring_ring(**self.ARGS)
        self.assertAlmostEqual(cc.hierarchical_all_gather_ring_ring(**self.ARGS), ar / 2)
        self.assertAlmostEqual(cc.hierarchical_reduce_scatter_ring_ring(**self.ARGS), ar / 2)

    def test_closed_form(self):
        a = self.ARGS
        n, l, m = a["N"], a["L"], a["M"]
        alpha = (n // l - 1) * a["alpha_inner"] + (l - 1) * a["alpha_outer"]
        bw = (n - l) / n * m / a["bw_inner"] + (l - 1) / n * m / a["bw_outer"]
        self.assertAlmostEqual(cc.hierarchical_all_gather_ring_ring(**a), alpha + bw)

    def test_degenerate_and_invalid(self):
        flat = cc.ring_all_reduce(M, 512, ALPHA, BW)
        self.assertEqual(cc.hierarchical_all_reduce_ring_ring(M, 512, 1, ALPHA, BW, 1e-6, 50e9), flat)
        for bad_l in (7, 0, 1024):
            with self.assertRaises(ValueError):
                cc.hierarchical_all_reduce_ring_ring(M, 512, bad_l, ALPHA, BW, 1e-6, 50e9)
            with self.assertRaises(ValueError):
                cc.hierarchical_all_gather_ring_ring(M, 512, bad_l, ALPHA, BW, 1e-6, 50e9)


class Contention(unittest.TestCase):
    def test_realistic_cost_flag_position_irrelevant(self):
        kw = cc.realistic_cost(cc.tree_all_reduce, M, G, ALPHA, BW, pipelined=True, eta_beta=0.8)
        pos = cc.realistic_cost(cc.tree_all_reduce, M, G, ALPHA, BW, True, eta_beta=0.8)
        self.assertEqual(kw, pos)
        self.assertAlmostEqual(us(kw), 53.444, places=2)

    def test_realistic_cost_rejects_per_tier_helpers(self):
        with self.assertRaises(TypeError):
            cc.realistic_cost(cc.hierarchical_all_reduce_ring_ring,
                              M, 512, 8, ALPHA, BW, 1e-6, 50e9, eta_beta=0.8)

    def test_profiles(self):
        self.assertEqual(ETA_PROFILES["crossbar"], (1.00, 0.80))
        self.assertEqual(ETA_PROFILES["nvls"], (1.00, 0.52))
        self.assertEqual(ETA_PROFILES["torus"], (1.20, 0.60))


class Guards(unittest.TestCase):
    def test_noop_group(self):
        self.assertEqual(cc.ring_all_reduce(M, 1, ALPHA, BW), 0.0)
        self.assertEqual(cc.torus_all_reduce(M, (1,), ALPHA, BW), 0.0)

    def test_zero_bandwidth_is_an_error(self):
        for call in (lambda: cc.ring_all_reduce(M, G, ALPHA, 0.0),
                     lambda: cc.p2p_hop(M, ALPHA, 0.0),
                     lambda: cc.torus_a2a(M, DIMS, ALPHA, -1.0)):
            with self.assertRaises(ValueError):
                call()

    def test_registry_runs(self):
        rows = run_all(Anchors())
        self.assertTrue(all(r["t_us"] > 0 for r in rows))
        self.assertEqual(len({r["name"] for r in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()
