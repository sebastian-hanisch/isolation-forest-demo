"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Datensätze belegt (Mittel; Toleranz ±0.02 = Rundung auf zwei Stellen plus Luft).
Positive UND negative Aussagen: wo der Isolation Forest verliert, steht das hier ebenso als Test wie dort, wo er gewinnt."""

from functools import lru_cache

import numpy as np
import pytest

import isf_constants as C
import isf_evaluation as ev

TOL = 0.02
STANDARD = ev.Settings()


@lru_cache(maxsize=None)
def _runs(items, settings):
    return tuple(ev.analyse(ev.make_dataset(seed=s, **dict(items)), settings) for s in C.SWEEP_SEEDS)


def runs(settings=STANDARD, **kw):
    return _runs(tuple(sorted(kw.items())), settings)


def m(det, key, settings=STANDARD, **kw):
    return float(np.nanmean([a.scores[det][key] for a in runs(settings, **kw)]))


def oracle(settings=STANDARD, **kw):
    return float(np.mean([a.oracle_f1 for a in runs(settings, **kw)]))


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Seitenleiste: Touren und Merkmale ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,fa,f1,cla_recall,rob_fa", [(20, 0.156, 0.60, 0.00, 0.16), (30, 0.119, 0.67, 0.00, 0.24), (50, 0.062, 0.78, 0.20, 0.29), (100, 0.027, 0.89, 0.42, 0.21), (200, 0.009, 0.96, 0.44, 0.06)])
def test_tours_sweep(n, fa, f1, cla_recall, rob_fa):
    near(m("iforest", "false_alarm", n=n), fa, 0.01)
    near(m("iforest", "f1", n=n), f1)
    near(m("iforest", "auc", n=n), 1.00, 0.01)
    near(m("classical", "recall", n=n), cla_recall)
    near(m("robust", "false_alarm", n=n), rob_fa, 0.015)


@pytest.mark.parametrize("p,f1,cla_recall,rob_fa", [(2, 0.92, 0.84, 0.03), (5, 0.92, 0.73, 0.02), (8, 0.94, 0.57, 0.03), (12, 0.96, 0.43, 0.04), (20, 0.96, 0.28, 0.08), (30, 0.95, 0.21, 0.16)])
def test_feature_count_sweep(p, f1, cla_recall, rob_fa):
    near(m("iforest", "auc", p=p), 1.00, 0.01)
    near(m("iforest", "f1", p=p), f1)
    near(m("classical", "recall", p=p), cla_recall)
    near(m("robust", "false_alarm", p=p), rob_fa, 0.015)


@pytest.mark.parametrize("nn,auc,recall,f1,rob_auc,cla_auc", [(0, 1.00, 0.99, 0.96, 1.00, 0.95), (10, 1.00, 0.91, 0.94, 0.99, 0.90), (20, 1.00, 0.71, 0.82, 0.96, 0.86), (30, 0.99, 0.58, 0.72, 0.91, 0.82),
                                                               (40, 0.99, 0.39, 0.56, 0.88, 0.79)])
def test_noise_features_keep_the_ranking_but_not_the_fixed_threshold(nn, auc, recall, f1, rob_auc, cla_auc):
    near(m("iforest", "auc", n_noise=nn), auc, 0.012)
    near(m("iforest", "recall", n_noise=nn), recall)
    near(m("iforest", "f1", n_noise=nn), f1)
    near(m("robust", "auc", n_noise=nn), rob_auc, 0.015)
    near(m("classical", "auc", n_noise=nn), cla_auc, 0.015)
    if nn == 40:
        near(oracle(n_noise=40), 0.84)                                                                    # mit bekanntem Anteil trägt die Rangfolge


@pytest.mark.parametrize("modes,if_recall,rob_recall,rob_auc", [(1, 0.99, 0.97, 1.00), (2, 0.99, 0.77, 0.95), (3, 0.97, 0.47, 0.85)])
def test_modes_sweep(modes, if_recall, rob_recall, rob_auc):
    near(m("iforest", "recall", n_modes=modes), if_recall)
    near(m("robust", "recall", n_modes=modes), rob_recall)
    near(m("robust", "auc", n_modes=modes), rob_auc)


@pytest.mark.parametrize("curv,if_f1,rob_f1,rob_fa,cla_f1", [(0.0, 0.96, 0.84, 0.04, 0.57), (0.25, 0.98, 0.49, 0.24, 0.90), (0.5, 0.97, 0.41, 0.32, 0.91), (0.75, 0.97, 0.39, 0.35, 0.91), (1.0, 0.95, 0.38, 0.36, 0.91)])
def test_curvature_sweep_the_ranking_is_the_same_for_all_only_the_chi2_threshold_suffers(curv, if_f1, rob_f1, rob_fa, cla_f1):
    near(m("iforest", "f1", curvature=curv), if_f1)
    near(m("robust", "f1", curvature=curv), rob_f1)
    near(m("robust", "false_alarm", curvature=curv), rob_fa, 0.015)
    near(m("classical", "f1", curvature=curv), cla_f1)
    for det in ev.DETECTORS:
        if curv > 0:
            near(m(det, "auc", curvature=curv), 1.00, 0.01)


@pytest.mark.parametrize("noise,if_recall,cla_recall", [(0.0, 0.99, 0.87), (0.25, 0.99, 0.43), (0.5, 0.97, 0.42), (1.0, 0.94, 0.38)])
def test_noise_sweep(noise, if_recall, cla_recall):
    near(m("iforest", "recall", noise=noise), if_recall)
    near(m("iforest", "auc", noise=noise), 1.00, 0.01)
    near(m("classical", "recall", noise=noise), cla_recall)


# --- Seitenleiste: Anomalien --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("c,if_f1,rob_f1,cla_recall", [(2, 0.42, 0.46, 0.93), (5, 0.81, 0.71, 0.75), (10, 0.96, 0.84, 0.43), (20, 0.97, 0.92, 0.14), (30, 0.93, 0.93, 0.06), (40, 0.87, 0.85, 0.04), (45, 0.81, 0.53, 0.04)])
def test_contamination_sweep(c, if_f1, rob_f1, cla_recall):
    near(m("iforest", "auc", contamination=c), 1.00, 0.01)
    near(m("iforest", "f1", contamination=c), if_f1)
    near(m("robust", "f1", contamination=c), rob_f1)
    near(m("classical", "recall", contamination=c), cla_recall)


def test_dense_group_the_root_is_better_up_to_25_percent_and_isolation_forest_at_30():
    for c, if_auc, rob_auc, cla_auc in ((10, 0.95, 1.00, 0.83), (20, 0.85, 1.00, 0.62), (30, 0.70, 0.49, 0.52), (40, 0.40, 0.39, 0.44)):
        near(m("iforest", "auc", kind="cluster", contamination=c), if_auc)
        near(m("robust", "auc", kind="cluster", contamination=c), rob_auc)
        near(m("classical", "auc", kind="cluster", contamination=c), cla_auc)
    assert m("robust", "auc", kind="cluster", contamination=20) > m("iforest", "auc", kind="cluster", contamination=20) + 0.1
    assert m("iforest", "auc", kind="cluster", contamination=30) > m("robust", "auc", kind="cluster", contamination=30) + 0.15
    near(m("iforest", "f1", kind="cluster", contamination=20), 0.44)
    near(m("iforest", "false_alarm", kind="cluster", contamination=20), 0.137, 0.02)


def test_gap_anomalies_are_barely_above_chance_for_the_isolation_forest_and_below_for_the_root():
    near(m("iforest", "auc", n_modes=2, kind="gap"), 0.54)
    near(m("classical", "auc", n_modes=2, kind="gap"), 0.40)
    near(m("robust", "auc", n_modes=2, kind="gap"), 0.40)
    near(m("iforest", "f1", n_modes=2, kind="gap"), 0.02, 0.03)
    near(m("iforest", "auc", n_modes=2, kind="gap", contamination=2), 0.81, 0.03)                       # wenige, verstreute Lücken-Anomalien findet er besser
    near(m("iforest", "auc", n_modes=3, kind="gap"), 0.27, 0.04)                                        # drei Betriebsarten: unter Raten


@pytest.mark.parametrize("strength,if_auc,if_f1,rob_auc,rob_recall,cla_auc,cla_recall", [(3.0, 0.96, 0.66, 0.80, 0.23, 0.79, 0.12), (4.0, 0.99, 0.85, 0.93, 0.62, 0.88, 0.24), (6.0, 1.00, 0.96, 1.00, 0.97, 0.95, 0.43),
                                                                                       (9.0, 1.00, 0.99, 1.00, 1.00, 0.97, 0.62), (12.0, 1.00, 1.00, 1.00, 1.00, 0.98, 0.68)])
def test_strength_sweep(strength, if_auc, if_f1, rob_auc, rob_recall, cla_auc, cla_recall):
    near(m("iforest", "auc", strength=strength), if_auc, 0.012)
    near(m("iforest", "f1", strength=strength), if_f1)
    near(m("robust", "auc", strength=strength), rob_auc, 0.015)
    near(m("robust", "recall", strength=strength), rob_recall)
    near(m("classical", "auc", strength=strength), cla_auc, 0.015)
    near(m("classical", "recall", strength=strength), cla_recall)


# --- Isolation Forest: Bäume, Unterstichprobe, Schwellen ---------------------------------------------------------------------------------


@pytest.mark.parametrize("trees,f1", [(10, 0.94), (25, 0.95), (50, 0.96), (100, 0.96), (200, 0.96), (500, 0.96)])
def test_tree_count_sweep(trees, f1):
    near(m("iforest", "f1", ev.Settings(n_trees=trees)), f1)
    near(m("iforest", "auc", ev.Settings(n_trees=trees)), 1.00, 0.01)


def test_forest_seed_spread_shrinks_with_more_trees():
    spread = {r["n_trees"]: r for r in ev.parameter_table()["spread"]}
    near(spread[10]["f1_std"], 0.037, 0.01)
    near(spread[50]["f1_std"], 0.015, 0.006)
    near(spread[200]["f1_std"], 0.009, 0.005)
    assert spread[10]["f1_std"] > spread[50]["f1_std"] > spread[200]["f1_std"] and max(r["auc_std"] for r in spread.values()) < 0.003


@pytest.mark.parametrize("psi,f1,fa", [(16, 0.56, 0.18), (32, 0.71, 0.10), (64, 0.83, 0.05), (128, 0.91, 0.02), (256, 0.96, 0.01)])
def test_psi_sweep_the_score_is_not_comparable_across_psi(psi, f1, fa):
    s = ev.Settings(psi=psi)
    near(m("iforest", "f1", s), f1)
    near(m("iforest", "false_alarm", s), fa, 0.015)
    near(m("iforest", "auc", s), 1.00, 0.012)


def test_small_psi_helps_against_masking():
    near(m("iforest", "auc", ev.Settings(psi=16), kind="cluster", contamination=30), 0.84)
    near(m("iforest", "auc", ev.Settings(psi=256), kind="cluster", contamination=30), 0.70)


@pytest.mark.parametrize("cut,f1,recall,fa", [(0.45, 0.82, 1.00, 0.050), (0.5, 0.96, 0.99, 0.009), (0.55, 0.97, 0.94, 0.001), (0.6, 0.83, 0.71, 0.000), (0.65, 0.55, 0.38, 0.000)])
def test_cutoff_sweep(cut, f1, recall, fa):
    s = ev.Settings(cutoff=cut)
    near(m("iforest", "f1", s), f1)
    near(m("iforest", "recall", s), recall)
    near(m("iforest", "false_alarm", s), fa, 0.01)


@pytest.mark.parametrize("q,rob_f1,cla_recall", [(0.9, 0.67, 0.77), (0.95, 0.77, 0.61), (0.975, 0.84, 0.43), (0.99, 0.89, 0.30), (0.999, 0.90, 0.08)])
def test_chi2_quantile_sweep_for_the_root_detectors(q, rob_f1, cla_recall):
    s = ev.Settings(quantile=q)
    near(m("robust", "f1", s), rob_f1)
    near(m("classical", "recall", s), cla_recall)
    near(m("iforest", "f1", s), m("iforest", "f1"), 1e-12)                                              # der Isolation Forest hängt nicht am chi²-Quantil


@pytest.mark.parametrize("share,if_f1,rob_f1,cla_f1", [(2, 0.33, 0.33, 0.32), (5, 0.67, 0.67, 0.56), (10, 0.97, 0.90, 0.69), (20, 0.67, 0.66, 0.58), (40, 0.40, 0.40, 0.39)])
def test_assumed_share_sweep(share, if_f1, rob_f1, cla_f1):
    s = ev.Settings(threshold_kind="share", share=share)
    near(m("iforest", "f1", s), if_f1)
    near(m("robust", "f1", s), rob_f1)
    near(m("classical", "f1", s), cla_f1)
    near(m("iforest", "auc", s), 1.00, 0.01)


def test_normal_scores_depend_on_the_number_of_tours():
    rows = {r["n"]: r["median"] for r in ev.threshold_table()["normal_scores"]}
    for n, expected in ((20, 0.435), (50, 0.401), (100, 0.382), (300, 0.376), (600, 0.377)):
        near(rows[n], expected, 0.012)
    assert rows[20] > rows[50] > rows[100] > rows[300] - 0.01


# --- Presets und Grenzen-Tabelle ---------------------------------------------------------------------------------------------------------


def test_preset_help_numbers():
    near(m("iforest", "f1"), 0.96)
    near(m("iforest", "false_alarm"), 0.009, 0.005)
    near(m("robust", "f1"), 0.84)
    near(m("robust", "false_alarm"), 0.039, 0.006)
    near(m("classical", "f1"), 0.57)
    near(m("classical", "auc"), 0.95, 0.01)
    p3 = dict(n=20, p=30)
    s3 = ev.Settings(psi=20)
    near(m("iforest", "auc", s3, **p3), 1.00, 0.01)
    near(m("classical", "auc", s3, **p3), 0.60)
    near(m("robust", "auc", s3, **p3), 0.55)
    assert m("classical", "recall", s3, **p3) == 0.0 and m("robust", "recall", s3, **p3) == 0.0 and m("classical", "n_flagged", s3, **p3) == 0.0
    near(m("iforest", "false_alarm", s3, **p3), 0.144, 0.02)
    near(m("iforest", "f1", s3, **p3), 0.61, 0.03)
    near(oracle(s3, **p3), 1.00, 0.01)
    share5 = ev.Settings(threshold_kind="share", share=5)
    near(m("iforest", "recall", share5), 0.50)
    near(oracle(share5), 0.97)


def test_ghost_anisotropy_is_positive_and_between_0_06_and_0_08_beyond_one_sigma():
    rows = ev.ghost_table()
    for r in rows:
        if r["distance"] >= 1.0:
            assert r["n_valid"] >= 4 and 0.055 <= r["mean"] <= 0.09 and r["min"] > 0.04
        else:
            assert r["mean"] <= 0.12
    by = {(r["n_modes"], r["distance"]): r["mean"] for r in rows}
    near(by[(1, 1.0)], 0.081, 0.012)
    near(by[(2, 2.0)], 0.061, 0.012)
    assert all(by[(k, 1.0)] > by[(k, 2.0)] for k in (1, 2, 3))                                         # nahe am Datenrand stärker


def test_dimension_table_the_isolation_forest_is_never_blind():
    cells = {(c["n"], c["p"]): c for c in ev.dimension_table()}
    assert min(c["iforest_auc"] for c in cells.values()) >= 0.985
    for key in ((20, 20), (20, 30), (30, 30)):
        assert cells[key]["robust_auc"] <= 0.56 and cells[key]["robust_auc"] >= 0.35
    near(cells[(20, 20)]["robust_auc"], 0.36, 0.03)
    near(cells[(30, 30)]["robust_auc"], 0.50, 0.03)
    assert cells[(400, 12)]["robust_auc"] > 0.98


def test_masking_and_modes_tables_have_the_documented_cells():
    rows = ev.modes_table()
    assert len(rows) == 8 and not any(r["n_modes"] == 1 and r["kind"] == "gap" for r in rows)
    by = {(r["n_modes"], r["kind"]): r for r in rows}
    near(by[(1, "cluster")]["robust_auc"], 1.00, 0.01)
    near(by[(1, "cluster")]["iforest_auc"], 0.95)
    near(by[(2, "cluster")]["iforest_auc"], 0.98)
    near(by[(2, "cluster")]["robust_auc"], 1.00, 0.01)
    near(by[(3, "cluster")]["iforest_auc"], 0.95)
    near(by[(3, "cluster")]["robust_auc"], 0.78)
    near(by[(3, "gap")]["iforest_auc"], 0.27, 0.04)
    mk = ev.masking_table()
    assert [r["x"] for r in mk["rows"]] == list(ev.MASKING_CONTAMINATION) and [r["x"] for r in mk["psi"]] == [16, 32, 64, 128, 256]
    assert mk["psi"][0]["iforest_auc"] > mk["psi"][-1]["iforest_auc"]


def test_threshold_table_shapes_and_wrong_share_factors():
    t = ev.threshold_table()
    assert [r["x"] for r in t["cutoff"]] == [0.45, 0.5, 0.55, 0.6, 0.65]
    assert [(r["factor"], r["x"]) for r in t["wrong_share"]] == [(0.5, 5), (1.0, 10), (2.0, 20)]
    f = {r["factor"]: r for r in t["wrong_share"]}
    near(f[1.0]["iforest_f1"], 0.97)
    near(f[0.5]["iforest_f1"], 0.67)
    near(f[2.0]["iforest_f1"], 0.67)
    assert f[1.0]["iforest_f1"] > f[1.0]["robust_f1"] > f[1.0]["classical_f1"]


def test_parameter_table_shapes():
    t = ev.parameter_table()
    assert [r["x"] for r in t["trees"]] == list(ev.SWEEP_VALUES["n_trees"]) and [r["x"] for r in t["psi"]] == list(ev.SWEEP_VALUES["psi"]) and [r["n_trees"] for r in t["spread"]] == [10, 50, 200]


def test_analysis_time_stays_small():
    a = ev.analyse(ev.make_dataset(n=600, p=30, n_noise=40, contamination=45))
    assert a.seconds["iforest"] < 3.0 and a.seconds["robust"] < 10.0
    b = ev.analyse(ev.make_dataset())
    assert b.seconds["iforest"] < 1.0
