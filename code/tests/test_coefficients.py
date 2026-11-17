"""Coefficient-level tests for ``core.collective_cost`` and the ``util`` layer.

Every cost in the model is ``t = n_α·α + n_β·M/BW``. Calling a function with
``(M=0, α=1, BW=1)`` returns ``n_α`` exactly and with ``(M=1, α=0, BW=1)``
returns ``n_β``, so each function is checked against the ``(n_α, n_β)`` pair
written in the specification (``documentation/modeling/00_summary.md`` §4-§7
lists them) at several group sizes, including ones that are not powers of two.
The expected values below are written out independently of the module.

Run from the repo root:

    .venv/bin/python -m unittest discover -s code/tests -v
"""

import dataclasses
import importlib.util
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import collective_cost as cc  # noqa: E402
from util import ETA_PROFILES, PRIMITIVES, Anchors, log_space, run_all, sweep, to_us  # noqa: E402
from util import style  # noqa: E402

GROUPS = (2, 3, 6, 72, 512)
SHAPES = ((3,), (7,), (5, 3), (16, 8, 4), (8, 8, 8), (2, 2, 2), (4, 1, 3))


def lg(n: int) -> int:
    """⌈log₂ n⌉ in integer arithmetic."""
    return (n - 1).bit_length()


def pair(fn, *group, **flags):
    """(n_α, n_β) of ``fn`` for the given group arguments."""
    return fn(0.0, *group, 1.0, 1.0, **flags), fn(1.0, *group, 0.0, 1.0, **flags)


class Close(unittest.TestCase):
    def close(self, got, want, msg=None):
        self.assertTrue(math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-15), msg or f"{got!r} != {want!r}")

    def pair_is(self, got, want, msg):
        self.close(got[0], want[0], f"n_α {msg}: {got[0]!r} != {want[0]!r}")
        self.close(got[1], want[1], f"n_β {msg}: {got[1]!r} != {want[1]!r}")


class StarCoefficients(Close):
    """01_collective_algorithms.md §3-§8 and App. A-B; 04_in_network_collectives.md §1."""

    def test_group_functions(self):
        table = {
            # name: (flags, n_α(G), n_β(G))
            "ring_broadcast": ({}, lambda g: g - 1, lambda g: g - 1),
            "ring_broadcast/p": ({"pipelined": True}, lambda g: g - 1, lambda g: 1),
            "tree_broadcast": ({}, lg, lg),
            "tree_broadcast/p": ({"pipelined": True}, lg, lambda g: 1),
            "ring_reduce": ({}, lambda g: g - 1, lambda g: g - 1),
            "ring_reduce/p": ({"pipelined": True}, lambda g: g - 1, lambda g: 1),
            "tree_reduce": ({}, lg, lg),
            "tree_reduce/p": ({"pipelined": True}, lg, lambda g: 1),
            "ring_all_reduce": ({}, lambda g: 2 * (g - 1), lambda g: 2 * (g - 1) / g),
            "tree_all_reduce": ({}, lambda g: 2 * lg(g), lambda g: 2 * lg(g)),
            "tree_all_reduce/p": ({"pipelined": True}, lambda g: 2 * lg(g), lambda g: 2),
            "tree_all_reduce/m": ({"multi_port": True}, lambda g: 2 * lg(g), lg),
            "tree_all_reduce/pm": ({"pipelined": True, "multi_port": True}, lambda g: 2 * lg(g), lambda g: 1),
            "rabenseifner_all_reduce": ({}, lambda g: 2 * lg(g), lambda g: 2 * (g - 1) / g),
            "ring_all_gather": ({}, lambda g: g - 1, lambda g: (g - 1) / g),
            "ring_reduce_scatter": ({}, lambda g: g - 1, lambda g: (g - 1) / g),
            "recursive_doubling_all_gather": ({}, lg, lambda g: (g - 1) / g),
            "recursive_halving_reduce_scatter": ({}, lg, lambda g: (g - 1) / g),
            "pat_all_gather": ({}, lg, lambda g: (g - 1) / g),
            "pat_reduce_scatter": ({}, lg, lambda g: (g - 1) / g),
            "inc_all_gather": ({}, lambda g: 2, lambda g: (g - 1) / g),
            "inc_reduce_scatter": ({}, lambda g: 2, lambda g: (g - 1) / g),
            "inc_a2a": ({}, lambda g: 2, lambda g: (g - 1) / g),
            "pairwise_a2a": ({}, lambda g: g - 1, lambda g: (g - 1) / g),
            "bruck_a2a": ({}, lg, lambda g: lg(g) / 2),
        }
        for name, (flags, n_alpha, n_beta) in table.items():
            fn = getattr(cc, name.split("/")[0])
            for g in GROUPS:
                self.pair_is(pair(fn, g, **flags), (n_alpha(g), n_beta(g)), f"{name} at G={g}")
            self.assertEqual(fn(1.0, 1, 1.0, 1.0, **flags), 0.0, f"{name} at G=1")

    def test_every_function_rejects_a_dead_link(self):
        import inspect
        skip = {"apply_eta", "realistic_cost", "canonical_anchors"}
        for name, fn in vars(cc).items():
            if name.startswith("_") or name in skip or not inspect.isfunction(fn):
                continue
            params = list(inspect.signature(fn).parameters)
            if "bw_Bps" not in params:
                continue                                   # the hierarchical helpers take per-tier values
            args = {"M": 1.0, "G": 8, "dims": (4, 2), "alpha_s": 1.0, "bw_Bps": 0.0}
            with self.assertRaises(ValueError, msg=name):
                fn(*[args[p] for p in params if p in args])

    def test_groupless_functions(self):
        self.pair_is((cc.p2p_hop(0.0, 1.0, 1.0), cc.p2p_hop(1.0, 0.0, 1.0)), (1, 1), "p2p_hop")
        for fn in (cc.inc_broadcast, cc.inc_reduce, cc.inc_all_reduce):
            self.pair_is((fn(0.0, 1.0, 1.0), fn(1.0, 0.0, 1.0)), (2, 1), fn.__name__)

    def test_all_reduce_is_reduce_scatter_plus_all_gather(self):
        m, a, bw = 16e6, 0.5e-6, 900e9
        for g in GROUPS:
            self.close(cc.ring_all_reduce(m, g, a, bw), cc.ring_reduce_scatter(m, g, a, bw) + cc.ring_all_gather(m, g, a, bw))
            self.close(cc.rabenseifner_all_reduce(m, g, a, bw),
                       cc.recursive_halving_reduce_scatter(m, g, a, bw) + cc.recursive_doubling_all_gather(m, g, a, bw))

    def test_ring_relay_busiest_link(self):
        """01 §7.1: with antipodal chunks split both ways the busiest link carries (G/8)·M."""
        def busiest(n):
            load = [0.0] * (2 * n)                       # [rightward links | leftward links]
            for s in range(n):
                for r in range(1, n):
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
            return max(load) / n                         # chunks of size M/n
        for n in (4, 8, 16, 64):
            self.close(cc.ring_relay_a2a(1.0, n, 0.0, 1.0), busiest(n))
            self.close(cc.ring_relay_a2a(0.0, n, 1.0, 1.0), n // 2)
        self.close(cc.ring_relay_a2a(1.0, 2, 0.0, 1.0), 0.5)   # one link, M/2 each way
        self.close(cc.ring_relay_a2a(0.0, 7, 1.0, 1.0), 3)


class TorusCoefficients(Close):
    """02_topology_mapping.md §3.2-§3.6 and §4.2."""

    def test_dimension_decomposed_rings(self):
        for dims in SHAPES:
            n = math.prod(dims)
            hops = sum(d - 1 for d in dims)
            self.pair_is(pair(cc.torus_all_reduce, dims), (2 * hops, 2 * (n - 1) / n), f"torus_all_reduce {dims}")
            self.pair_is(pair(cc.torus_all_gather, dims), (hops, (n - 1) / n), f"torus_all_gather {dims}")
            self.pair_is(pair(cc.torus_reduce_scatter, dims), (hops, (n - 1) / n), f"torus_reduce_scatter {dims}")

    def test_all_to_all(self):
        for dims in SHAPES:
            torus = max(d / 8 if d >= 3 else d / 4 for d in dims if d > 1)   # a size-2 dimension has no wraparound link
            self.pair_is(pair(cc.torus_a2a, dims), (sum(d // 2 for d in dims), torus), f"torus_a2a {dims}")
            self.pair_is(pair(cc.torus_a2a, dims, wraparound=False),
                         (sum(d - 1 for d in dims), max(dims) / 4), f"mesh a2a {dims}")

    def test_broadcast_and_reduce(self):
        for fn in (cc.torus_broadcast, cc.torus_reduce):
            for dims in SHAPES:
                ring, line = sum(d // 2 for d in dims), sum(d - 1 for d in dims)
                self.pair_is(pair(fn, dims), (ring, ring), f"{fn.__name__} {dims}")
                self.pair_is(pair(fn, dims, pipelined=True), (ring, 1), f"{fn.__name__} pipelined {dims}")
                self.pair_is(pair(fn, dims, wraparound=False), (line, line), f"{fn.__name__} mesh {dims}")
                self.pair_is(pair(fn, dims, wraparound=False, pipelined=True), (line, 1), f"{fn.__name__} mesh pipelined {dims}")

    def test_one_dimension_is_a_ring(self):
        m, a, bw = 16e6, 0.5e-6, 900e9
        for g in (3, 6, 72):
            self.close(cc.torus_all_reduce(m, (g,), a, bw), cc.ring_all_reduce(m, g, a, bw))
            self.close(cc.torus_all_gather(m, (g,), a, bw), cc.ring_all_gather(m, g, a, bw))


class NoOpGroups(unittest.TestCase):
    """A group of one rank (or none) exchanges nothing."""

    def test_torus_and_hierarchy(self):
        for dims in ((), (1,), (1, 1, 1)):
            for fn in (cc.torus_all_reduce, cc.torus_all_gather, cc.torus_reduce_scatter, cc.torus_a2a,
                       cc.torus_broadcast, cc.torus_reduce):
                self.assertEqual(fn(1.0, dims, 1.0, 1.0), 0.0, f"{fn.__name__} {dims}")
        for fn in (cc.hierarchical_all_reduce_ring_ring, cc.hierarchical_all_gather_ring_ring,
                   cc.hierarchical_reduce_scatter_ring_ring):
            for n in (0, 1):
                self.assertEqual(fn(1.0, n, 1, 1.0, 1.0, 1.0, 1.0), 0.0, fn.__name__)
            for bad in (0.5, -4, float("inf")):
                with self.assertRaises(ValueError):
                    fn(1.0, bad, 1, 1.0, 1.0, 1.0, 1.0)

    def test_non_finite_sizes_are_value_errors(self):
        inf, nan = float("inf"), float("nan")
        for call in (lambda: cc.ring_all_reduce(1.0, inf, 1.0, 1.0), lambda: cc.ring_all_reduce(1.0, nan, 1.0, 1.0),
                     lambda: cc.torus_all_reduce(1.0, (inf, 2), 1.0, 1.0), lambda: cc.apply_eta(1.0, 1.0, 0.0, 1.0)):
            with self.assertRaises(ValueError):
                call()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "array payloads need numpy")
    def test_array_payloads(self):
        import numpy as np
        m = np.array([0.0, 1.0, 2.0])
        got = cc.ring_all_reduce(m, 4, 1.0, 1.0)
        self.assertEqual(got.tolist(), [6.0, 7.5, 9.0])                    # 2(N-1) + 2(N-1)/N · M
        self.assertEqual(cc.torus_all_reduce(1.0, np.array([8, 8, 8]), 1.0, 1.0), cc.torus_all_reduce(1.0, (8, 8, 8), 1.0, 1.0))
        with self.assertRaises(ValueError):
            cc.ring_all_reduce(np.array([1.0, -1.0]), 4, 1.0, 1.0)


class HierarchyCoefficients(Close):
    """03_hierarchical_topologies.md §2.1."""

    TIERS = (0.5e-6, 900e9, 2e-6, 100e9)

    def test_closed_forms(self):
        m = 16e6
        ai, bi, ao, bo = self.TIERS
        for n, l in ((144, 2), (512, 8), (512, 64), (288, 4), (12, 3)):
            alpha = (n // l - 1) * ai + (l - 1) * ao
            bw = (n - l) / n * m / bi + (l - 1) / n * m / bo
            self.close(cc.hierarchical_all_reduce_ring_ring(m, n, l, *self.TIERS), 2 * (alpha + bw))
            self.close(cc.hierarchical_all_gather_ring_ring(m, n, l, *self.TIERS), alpha + bw)
            self.close(cc.hierarchical_reduce_scatter_ring_ring(m, n, l, *self.TIERS), alpha + bw)

    def test_one_pod_and_one_rank_per_pod(self):
        m, n = 16e6, 512
        ai, bi, ao, bo = self.TIERS
        for hier, flat in ((cc.hierarchical_all_reduce_ring_ring, cc.ring_all_reduce),
                           (cc.hierarchical_all_gather_ring_ring, cc.ring_all_gather),
                           (cc.hierarchical_reduce_scatter_ring_ring, cc.ring_reduce_scatter)):
            self.close(hier(m, n, 1, *self.TIERS), flat(m, n, ai, bi))      # one pod: the inner tier only
            self.close(hier(m, n, n, *self.TIERS), flat(m, n, ao, bo))      # one rank per pod: the outer tier only

    def test_invalid_pod_count(self):
        for hier in (cc.hierarchical_all_reduce_ring_ring, cc.hierarchical_all_gather_ring_ring,
                     cc.hierarchical_reduce_scatter_ring_ring):
            for bad in (7, 0, -2, 1024):
                with self.assertRaises(ValueError):
                    hier(16e6, 512, bad, *self.TIERS)

    def test_generic_composers_add_their_phases(self):
        self.assertEqual(cc.hierarchical_all_reduce(1.0, 2.0, 4.0), 7.0)
        self.assertEqual(cc.hierarchical_all_gather(1.0, 2.0), 3.0)
        self.assertEqual(cc.hierarchical_reduce_scatter(1.0, 2.0), 3.0)


class Contention(Close):
    """05_contention_and_congestion.md §4: t = η_α·n_α·α + n_β·M/(η_β·BW)."""

    def test_realistic_cost_scales_each_term(self):
        m, g, a, bw = 16e6, 72, 0.5e-6, 900e9
        for fn, n_alpha, n_beta in ((cc.ring_all_reduce, 2 * 71, 2 * 71 / 72), (cc.pairwise_a2a, 71, 71 / 72)):
            got = cc.realistic_cost(fn, m, g, a, bw, eta_alpha=1.2, eta_beta=0.6)
            self.close(got, 1.2 * n_alpha * a + n_beta * m / (0.6 * bw))
        got = cc.realistic_cost(cc.torus_a2a, m, (8, 4), a, bw, False, eta_alpha=1.1, eta_beta=0.5)
        self.close(got, 1.1 * 10 * a + 2 * m / (0.5 * bw))
        self.assertEqual(cc.apply_eta(2.0, 10.0, 1.5, 0.5), (3.0, 5.0))


class PinnedAnchors(unittest.TestCase):
    """The seven regression anchors, with the values and the tolerance written here and not read from the module."""

    TARGETS_US = (546.49, 44.56, 18.78, 56.49, 35.19, 53.44, 84.34)

    def test_values(self):
        got = [value for _label, value, _target in cc.canonical_anchors()]
        self.assertEqual(len(got), len(self.TARGETS_US))
        for value, target in zip(got, self.TARGETS_US):
            self.assertAlmostEqual(value, target, delta=0.005)

    def test_module_targets_and_tolerance(self):
        self.assertEqual(tuple(target for _l, _v, target in cc.canonical_anchors()), self.TARGETS_US)
        self.assertEqual(cc.ANCHOR_TOLERANCE_US, 0.005)


class Registry(Close):
    """Every row of ``util.registry`` equals the closed form it is labelled with, at the default anchor."""

    R = 511 / 512
    ROWS = {
        # name: (n_α, n_β) at G = 512, dims = (8, 8, 8)
        "p2p_hop": (1, 1),
        "ring_broadcast": (511, 511), "ring_broadcast (pipelined)": (511, 1),
        "tree_broadcast": (9, 9), "tree_broadcast (pipelined)": (9, 1),
        "inc_broadcast": (2, 1),
        "torus_broadcast": (12, 12), "torus_broadcast (pipelined)": (12, 1), "torus_broadcast (mesh)": (21, 21),
        "ring_reduce": (511, 511), "ring_reduce (pipelined)": (511, 1),
        "tree_reduce": (9, 9), "tree_reduce (pipelined)": (9, 1),
        "inc_reduce": (2, 1),
        "torus_reduce": (12, 12), "torus_reduce (pipelined)": (12, 1),
        "ring_all_reduce": (1022, 2 * R),
        "tree_all_reduce": (18, 18), "tree_all_reduce (pipelined)": (18, 2),
        "tree_all_reduce (pipelined, multi-port)": (18, 1),
        "rabenseifner_all_reduce": (18, 2 * R), "inc_all_reduce": (2, 1), "torus_all_reduce": (42, 2 * R),
        "ring_all_gather": (511, R), "recursive_doubling_all_gather": (9, R), "pat_all_gather": (9, R),
        "inc_all_gather": (2, R), "torus_all_gather": (21, R),
        "ring_reduce_scatter": (511, R), "recursive_halving_reduce_scatter": (9, R), "pat_reduce_scatter": (9, R),
        "inc_reduce_scatter": (2, R), "torus_reduce_scatter": (21, R),
        "pairwise_a2a": (511, R), "ring_relay_a2a": (256, 64), "bruck_a2a": (9, 4.5), "inc_a2a": (2, R),
        "torus_a2a": (12, 1), "torus_a2a (mesh)": (21, 2),
    }

    def test_rows(self):
        a = Anchors()
        rows = {r["name"]: r["t_us"] for r in run_all(a)}
        self.assertEqual(len(rows), len(PRIMITIVES), "row names must be unique")
        hier = (a.n_hier // a.L - 1) * a.alpha_inner + (a.L - 1) * a.alpha_outer \
            + (a.n_hier - a.L) / a.n_hier * a.M / a.bw_inner + (a.L - 1) / a.n_hier * a.M / a.bw_outer
        want = {name: n_alpha * a.alpha + n_beta * a.M / a.bw for name, (n_alpha, n_beta) in self.ROWS.items()}
        want.update({"hierarchical_all_reduce_ring_ring": 2 * hier, "hierarchical_all_gather_ring_ring": hier,
                     "hierarchical_reduce_scatter_ring_ring": hier})
        self.assertEqual(set(rows), set(want))
        for name, seconds in want.items():
            self.close(rows[name], seconds * 1e6, f"{name}: {rows[name]!r} != {seconds * 1e6!r}")

    def test_every_cost_function_is_registered(self):
        skip = {"apply_eta", "realistic_cost", "canonical_anchors",
                "hierarchical_all_reduce", "hierarchical_all_gather", "hierarchical_reduce_scatter"}
        public = {n for n, f in vars(cc).items() if callable(f) and not n.startswith("_") and f.__module__ == cc.__name__}
        registered = {spec.name.split(" (")[0] for spec in PRIMITIVES}
        self.assertEqual(public - skip, registered)

    def test_anchor_defaults(self):
        a = Anchors()
        self.assertEqual((a.M, a.G, a.alpha, a.bw, a.dims), (16e6, 512, 0.5e-6, 900e9, (8, 8, 8)))
        self.assertEqual((a.L, a.N_hier, a.n_hier, a.alpha_inner, a.bw_inner, a.alpha_outer, a.bw_outer),
                         (8, 0, 512, 0.5e-6, 900e9, 2.0e-6, 100e9))
        self.assertEqual(to_us(1.5e-6), 1.5)
        self.assertEqual(ETA_PROFILES["ideal"], (1.00, 1.00))

    def test_anchor_overrides_stay_consistent(self):
        self.assertEqual(Anchors(G=64, dims=(4, 4, 4)).n_hier, 64)
        self.assertEqual(dataclasses.replace(Anchors(), G=64, dims=(4, 4, 4)).n_hier, 64)
        self.assertEqual(Anchors(N_hier=144, L=2).n_hier, 144)
        with self.assertRaises(ValueError):
            Anchors(G=64)                      # dims still describe 512 ranks
        with self.assertRaises(ValueError):
            Anchors(G=100, dims=(10, 10))      # L = 8 does not divide 100
        with self.assertRaises(ValueError):
            dataclasses.replace(Anchors(), G=100, dims=(10, 10))
        a = Anchors()
        a.G, a.dims = 64, (4, 4, 4)
        self.assertEqual(a.n_hier, 64)
        a.G = 100                              # changed after construction: run_all re-validates
        with self.assertRaises(ValueError):
            run_all(a)


class Helpers(unittest.TestCase):
    def test_log_space(self):
        xs = log_space(1e3, 1e9, 61)
        self.assertEqual((len(xs), xs[0], xs[-1]), (61, 1e3, 1e9))
        self.assertAlmostEqual(xs[10] / xs[0], 10.0, places=9)
        self.assertAlmostEqual(xs[31] / xs[30], xs[1] / xs[0], places=9)
        self.assertEqual(log_space(5.0, 50.0, 1), [5.0])
        for bad in ((0.0, 1.0, 3), (1.0, -1.0, 3), (1.0, 2.0, 0)):
            with self.assertRaises(ValueError):
                log_space(*bad)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "sweep() returns a pandas DataFrame")
    def test_sweep(self):
        df = sweep({"a": lambda x: 2e-6 * x, "b": lambda x: 1e-6}, [1.0, 3.0], x_name="M_bytes")
        self.assertEqual(list(df.columns), ["M_bytes", "label", "t_us"])
        self.assertEqual(df["t_us"].tolist(), [2.0, 1.0, 6.0, 1.0])       # seconds in, microseconds out
        self.assertEqual(df["label"].tolist(), ["a", "b", "a", "b"])
        for clash in ("label", "t_us"):
            with self.assertRaises(ValueError):
                sweep({"a": lambda x: x}, [1.0], x_name=clash)

    def test_style_family(self):
        cases = {"ring AR": "ring", "pairwise A2A": "ring", "DBT AR (pipelined)": "tree", "rec-doubling / PAT AG": "tree",
                 "Bruck A2A": "tree", "INC AR": "inc", "star + NVLS": "inc", "hardware A2A": "inc",
                 "Rabenseifner AR": "rabenseifner", "torus 8³ dim-ring": "torus"}
        for label, want in cases.items():
            self.assertEqual(style.family(label), want, label)
        with self.assertRaises(ValueError):
            style.family("no algorithm here")


if __name__ == "__main__":
    unittest.main()
