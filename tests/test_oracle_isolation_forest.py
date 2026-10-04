"""Unabhängige Orakel für Isolation Forest, die kopierten Elliptic-Envelope-Schätzer und die Kennzahlen.

- Erwartete Pfadlänge je Punkt EXAKT durch Aufzählung aller Schnittlücken (Wahrscheinlichkeit = Lückenbreite / Spannweite,
  Merkmal gleichverteilt unter den nicht konstanten) gegen den Mittelwert vieler gebauter Bäume.
- Baumstruktur: Größen durch erneutes Routen der Unterstichprobe, Höhenlimit, keine vorzeitigen Blätter, Pfadlänge = Tiefe + c(Blattgröße).
- Kennzahlen gegen scikit-learn (AUC, mittlere Präzision, Precision/Recall/F1) - auch bei Bindungen im Score.
- Mahalanobis-Abstand gegen die Pseudoinverse, MCD gegen vollständige Aufzählung aller h-Teilmengen (kleines n).
"""

import itertools
import math

import numpy as np
import pytest

import isf_algorithm as isf
import isf_ee_algorithm as alg
import isf_evaluation as ev

metrics = pytest.importorskip("sklearn.metrics")


def _c(m):
    return 0.0 if m <= 1 else (1.0 if m == 2 else 2 * (math.log(m - 1) + 0.5772156649015329) - 2 * (m - 1) / m)


def _expected_path(S, depth, limit):
    """Exakte erwartete Pfadlänge aller Zeilen von S an einem Knoten der Tiefe `depth` (paarweise verschiedene Werte je Merkmal)."""
    m, p = S.shape
    valid = [f for f in range(p) if S[:, f].max() > S[:, f].min()]
    if m == 1 or depth == limit or not valid:
        return np.full(m, depth + _c(m))
    out = np.zeros(m)
    for f in valid:
        order = np.argsort(S[:, f])
        xs = S[order, f]
        for i in range(1, m):
            gap = xs[i] - xs[i - 1]
            if gap <= 0:
                continue
            res = np.zeros(m)
            res[order[:i]] = _expected_path(S[order[:i]], depth + 1, limit)
            res[order[i:]] = _expected_path(S[order[i:]], depth + 1, limit)
            out += gap / (xs[-1] - xs[0]) * res / len(valid)
    return out


@pytest.mark.parametrize("seed,psi,p", [(0, 5, 1), (1, 6, 2), (2, 4, 2), (3, 7, 1)])
def test_mean_path_length_equals_the_exact_expectation(seed, psi, p):
    S = np.random.default_rng(seed).normal(size=(psi, p))
    limit = isf.height_limit(psi)
    rng = np.random.default_rng(100 + seed)
    mean = np.mean([isf.tree_path_length(isf.build_tree(S, rng, limit), S) for _ in range(6000)], axis=0)
    assert np.abs(mean - _expected_path(S, 0, limit)).max() < 0.05


def test_tree_structure_against_rerouting_and_path_length_formula():
    rng = np.random.default_rng(7)
    for it in range(60):
        psi, p = int(rng.integers(2, 60)), int(rng.integers(1, 5))
        S = rng.normal(size=(psi, p))
        if it % 4 == 0:
            S[: psi // 2] = S[0]
        if it % 5 == 0:
            S[:, 0] = 1.0
        limit = isf.height_limit(psi)
        t = isf.build_tree(S, np.random.default_rng(it), limit)
        sizes = np.zeros(len(t.feature), int)
        leaf_of = []
        for i in range(psi):
            node = 0
            sizes[0] += 1
            depth = 0
            while t.feature[node] >= 0:
                node = t.left[node] if S[i, t.feature[node]] < t.threshold[node] else t.right[node]
                sizes[node] += 1
                depth += 1
            leaf_of.append((node, depth))
        assert np.array_equal(sizes, t.size)
        assert t.depth.max() <= limit
        for node in range(len(t.feature)):
            if t.feature[node] >= 0:
                assert t.size[t.left[node]] + t.size[t.right[node]] == t.size[node]
            elif t.size[node] > 1 and t.depth[node] < limit:
                pts = [i for i in range(psi) if leaf_of[i][0] == node]
                assert (S[pts] == S[pts[0]]).all()          # nur gleiche Punkte bleiben vor dem Höhenlimit zusammen
        expect = np.array([d + _c(int(t.size[nd])) for nd, d in leaf_of])
        assert np.allclose(isf.tree_path_length(t, S), expect)


def test_c_factor_and_score_formula():
    ns = np.arange(1, 600)
    assert np.allclose(isf.c_factor(ns.astype(float)), [_c(int(n)) for n in ns], atol=1e-12)
    paths = np.random.default_rng(0).uniform(1, 9, size=(20, 15))
    assert np.allclose(isf.score_from_paths(paths, 64), 2.0 ** (-paths.mean(axis=0) / _c(64)))


def test_metrics_against_scikit_learn_including_tied_scores():
    rng = np.random.default_rng(1)
    for it in range(150):
        n = int(rng.integers(6, 80))
        s = rng.normal(size=n)
        if it % 2 == 0:
            s = np.round(s, 1)                                  # viele Bindungen
        y = rng.random(n) < rng.uniform(0.1, 0.6)
        if y.all() or not y.any():
            continue
        assert ev.roc_auc(s, y) == pytest.approx(metrics.roc_auc_score(y, s), abs=1e-12)
        assert ev.average_precision(s, y) == pytest.approx(metrics.average_precision_score(y, s), abs=1e-12)
        fpr, tpr = ev.roc_curve(s, y)
        f2, t2, _ = metrics.roc_curve(y, s, drop_intermediate=False)
        assert np.trapezoid(tpr, fpr) == pytest.approx(np.trapezoid(t2, f2), abs=1e-12)
        flagged = isf.flag_top(s, max(1, int(round(rng.uniform(0.05, 0.5) * n))) / n)
        m = ev.flag_metrics(flagged, y)
        p_, r_, f_, _ = metrics.precision_recall_fscore_support(y, flagged, average="binary", zero_division=0)
        assert (m["precision"], m["recall"], m["f1"]) == pytest.approx((p_, r_, f_), abs=1e-12)
        assert m["false_alarm"] == pytest.approx((flagged & ~y).sum() / (~y).sum())


def test_average_precision_does_not_depend_on_the_row_order_of_tied_scores():
    s, y = np.array([1.0, 1.0, 1.0, 1.0]), np.array([True, False, False, False])
    assert ev.average_precision(s, y) == pytest.approx(0.25)
    assert ev.average_precision(s[::-1], y[::-1]) == pytest.approx(0.25)


def test_classical_fit_and_mahalanobis_against_pseudoinverse():
    rng = np.random.default_rng(3)
    for _ in range(60):
        n, p = int(rng.integers(8, 60)), int(rng.integers(1, 6))
        X = rng.normal(size=(n, p)) @ rng.normal(size=(p, p)) + 3 * rng.normal(size=p)
        f = alg.fit_classical(X)
        assert np.allclose(f.location, X.mean(axis=0))
        assert np.allclose(f.covariance, np.atleast_2d(np.cov(X.T)))
        P = np.linalg.pinv(f.covariance)
        assert np.allclose(f.d2, [(x - f.location) @ P @ (x - f.location) for x in X], rtol=1e-6, atol=1e-8)
    X = rng.normal(size=(40, 3))
    X = np.column_stack([X, X[:, 0] + X[:, 1]])                  # Rang 3 statt 4
    f = alg.fit_classical(X)
    P = np.linalg.pinv(f.covariance, rcond=1e-10)
    assert f.dof == 3 and np.allclose(f.d2, [(x - f.location) @ P @ (x - f.location) for x in X], rtol=1e-5, atol=1e-7)


def test_mcd_support_against_exhaustive_search_on_tiny_data():
    rng = np.random.default_rng(4)
    for it in range(30):
        n, p = int(rng.integers(8, 11)), int(rng.integers(1, 3))
        X = rng.normal(size=(n, p))
        if it % 2 == 0:
            X[: int(rng.integers(1, 4))] += rng.uniform(4, 10)
        h = alg.subset_size(n, p, 0.5)

        def logdet(sub):
            return np.linalg.slogdet(np.atleast_2d(np.cov(X[list(sub)].T)))[1]

        best = min(logdet(sub) for sub in itertools.combinations(range(n), h))
        fit = alg.fit_mcd(X, 0.5, it, reweight=False)
        assert len(set(fit.support.tolist())) == h
        assert logdet(fit.support) >= best - 1e-9                 # kein Ergebnis kann das Optimum unterbieten
        assert logdet(fit.support) <= best + 1e-6                 # und FastMCD findet es hier
        assert fit.logdet_history[-1] == pytest.approx(logdet(fit.support), abs=1e-3)
