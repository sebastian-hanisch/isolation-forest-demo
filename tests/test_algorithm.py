"""Isolation Forest: c(n), Baumstruktur (Handinstanzen), Pfadlängen, Score gegen scikit-learn, Invarianzen, Determinismus; die kopierten Schätzer der Wurzel (χ² gegen scipy, MCD gegen scikit-learn)."""

import numpy as np
import pytest
from scipy.stats import chi2, spearmanr
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest

import isf_algorithm as isf
import isf_ee_algorithm as alg
import isf_evaluation as ev


def _data(n=300, p=4, n_out=15, shift=8.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.standard_normal((n - n_out, p)), shift + 0.5 * rng.standard_normal((n_out, p))])
    return X, np.arange(n) >= n - n_out


# --- c(n) ---------------------------------------------------------------------------------------------------------------------------------


def test_c_factor_definition_and_edge_values():
    assert isf.c_factor(np.array([1, 2])).tolist() == [0.0, 1.0]
    n = np.array([3.0, 10.0, 256.0])
    assert np.allclose(isf.c_factor(n), 2 * (np.log(n - 1) + isf.EULER_GAMMA) - 2 * (n - 1) / n)
    m = np.array([10.0, 50.0, 256.0])
    exact = 2 * np.array([sum(1 / k for k in range(1, int(x))) for x in m]) - 2 * (m - 1) / m                  # mit der exakten harmonischen Summe (ln + Euler-Konstante nähert sie an)
    assert np.allclose(isf.c_factor(m), exact, atol=0.15)
    assert (np.diff(isf.c_factor(np.arange(2, 400))) > 0).all()


def test_height_limit():
    assert [isf.height_limit(k) for k in (2, 16, 17, 256, 300)] == [1, 4, 5, 8, 9]


# --- Baum ---------------------------------------------------------------------------------------------------------------------------------


def _route(tree, X):
    """Blatt jedes Punkts (zur Kontrolle der Knotengrößen)."""
    node = np.zeros(len(X), dtype=int)
    for _ in range(64):
        f = tree.feature[node]
        internal = f >= 0
        if not internal.any():
            break
        go = X[np.arange(len(X)), np.maximum(f, 0)] < tree.threshold[node]
        node = np.where(internal, np.where(go, tree.left[node], tree.right[node]), node)
    return node


def test_tree_structure_sizes_thresholds_and_depth():
    X, _ = _data(n=64, n_out=4)
    tree = isf.build_tree(X, np.random.default_rng(1), isf.height_limit(64))
    assert tree.size[0] == 64 and tree.depth.max() <= isf.height_limit(64)
    for node in np.flatnonzero(tree.feature >= 0):
        assert tree.size[node] == tree.size[tree.left[node]] + tree.size[tree.right[node]] and tree.size[tree.left[node]] > 0 and tree.size[tree.right[node]] > 0
        assert tree.depth[tree.left[node]] == tree.depth[node] + 1
    leaves = _route(tree, X)
    assert np.array_equal(np.bincount(leaves, minlength=len(tree.size))[np.flatnonzero(tree.feature < 0)], tree.size[np.flatnonzero(tree.feature < 0)])     # Blattgrößen = wirklich dort landende Punkte
    # Schnittpunkt liegt im Wertebereich des Knotens
    for node in np.flatnonzero(tree.feature >= 0):
        reach = X[_reach(tree, X, node)]
        f = tree.feature[node]
        assert reach[:, f].min() <= tree.threshold[node] <= reach[:, f].max()


def _reach(tree, X, target):
    """Punkte, die den Knoten `target` durchlaufen."""
    mask = np.zeros(len(X), dtype=bool)
    for i, x in enumerate(X):
        node = 0
        while True:
            if node == target:
                mask[i] = True
                break
            if tree.feature[node] < 0:
                break
            node = tree.left[node] if x[tree.feature[node]] < tree.threshold[node] else tree.right[node]
    return mask


def test_full_depth_tree_isolates_every_point_and_constant_features_are_never_split():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.standard_normal(20), np.full(20, 5.0)])                          # zweite Spalte konstant
    tree = isf.build_tree(X, rng, max_depth=40)
    assert set(tree.feature[tree.feature >= 0]) == {0}
    assert (tree.size[tree.feature < 0] == 1).all()                                            # ohne Höhenlimit steht jeder Punkt allein


def test_identical_points_stay_in_one_leaf():
    tree = isf.build_tree(np.ones((10, 3)), np.random.default_rng(0), 5)
    assert len(tree.feature) == 1 and tree.feature[0] == -1 and tree.size[0] == 10
    assert isf.tree_path_length(tree, np.ones((2, 3)))[0] == pytest.approx(float(isf.c_factor(10)))


def test_path_length_equals_the_explicit_walk():
    X, _ = _data(n=100)
    tree = isf.build_tree(X, np.random.default_rng(2), isf.height_limit(100))
    lengths = isf.tree_path_length(tree, X)
    for i in (0, 10, 55, 99):
        steps, size = isf.point_path(tree, X[i])
        assert lengths[i] == pytest.approx(len(steps) + float(isf.c_factor(size)))
    assert lengths.min() >= 1 and lengths.max() <= isf.height_limit(100) + float(isf.c_factor(100))


# --- Wald und Score ------------------------------------------------------------------------------------------------------------------------


def test_scores_are_in_the_unit_interval_and_anomalies_score_higher():
    X, an = _data()
    F = isf.fit_forest(X, 100, 256, 0)
    s = isf.score(F, X)
    assert ((s > 0) & (s < 1)).all() and s[an].min() > s[~an].mean() and ev.roc_auc(s, an) > 0.99
    far = isf.score(F, np.full((1, 4), 40.0))[0]
    assert far > s.max() - 1e-9 or far > 0.7                                                    # ein Punkt weit außerhalb ist fast maximal auffällig


def test_score_identity_half_at_the_mean_path_c_psi():
    paths = np.full((5, 3), float(isf.c_factor(256)))
    assert isf.score_from_paths(paths, 256) == pytest.approx([0.5, 0.5, 0.5])
    assert isf.score_from_paths(np.zeros((2, 1)), 256)[0] == pytest.approx(1.0)


def test_psi_is_capped_by_n_and_forest_shape():
    X, _ = _data(n=50, n_out=3)
    F = isf.fit_forest(X, 7, 256, 0)
    assert F.psi == 50 and len(F.trees) == 7 and all(len(s) == 50 for s in F.subsets) and F.max_depth == isf.height_limit(50)
    assert isf.path_lengths(F, X).shape == (7, 50)
    G = isf.fit_forest(X, 3, 20, 0)
    assert G.psi == 20 and all(len(set(s.tolist())) == 20 for s in G.subsets)                 # ohne Zurücklegen


def test_determinism_and_seed_dependence():
    X, _ = _data()
    a, b, c = isf.score(isf.fit_forest(X, 50, 128, 3), X), isf.score(isf.fit_forest(X, 50, 128, 3), X), isf.score(isf.fit_forest(X, 50, 128, 4), X)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and spearmanr(a, c)[0] > 0.9


def test_score_converges_with_more_trees():
    X, an = _data()
    F = isf.fit_forest(X, 300, 256, 0)
    paths = isf.path_lengths(F, X)
    conv = isf.score_convergence(paths, F.psi, [1, 10, 100, 300])
    assert np.allclose(conv[-1], isf.score_from_paths(paths, F.psi))
    err = [np.abs(conv[k] - conv[-1]).mean() for k in range(3)]
    assert err[0] > err[1] > err[2]


def test_close_to_scikit_learn():
    for seed in (0, 1, 2):
        X, an = _data(seed=seed)
        ours = isf.score(isf.fit_forest(X, 100, 256, 0), X)
        sk = IsolationForest(n_estimators=100, max_samples=256, random_state=0).fit(X)
        theirs = -sk.score_samples(X)
        assert spearmanr(ours, theirs)[0] > 0.9
        assert abs(ev.roc_auc(ours, an) - ev.roc_auc(theirs, an)) < 0.02
        assert abs(float(ours.mean()) - float(theirs.mean())) < 0.03                          # gleiche Skala des Scores


def test_close_to_scikit_learn_on_the_scenario():
    ds = ev.make_dataset(seed=100000)
    ours = isf.score(isf.fit_forest(ds.X, 100, 256, 0), ds.X)
    theirs = -IsolationForest(n_estimators=100, max_samples=256, random_state=0).fit(ds.X).score_samples(ds.X)
    assert spearmanr(ours, theirs)[0] > 0.93 and abs(ev.roc_auc(ours, ds.anomaly) - ev.roc_auc(theirs, ds.anomaly)) < 0.02


def test_scale_and_shift_invariance_but_not_rotation_invariance():
    X, _ = _data(p=3)
    base = isf.score(isf.fit_forest(X, 60, 128, 5), X)
    a = np.array([1e-3, 5.0, 300.0])
    scaled = isf.score(isf.fit_forest(X * a + np.array([7.0, -3.0, 100.0]), 60, 128, 5), X * a + np.array([7.0, -3.0, 100.0]))
    assert np.allclose(base, scaled, atol=1e-9)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rotated = isf.score(isf.fit_forest(X @ R, 60, 128, 5), X @ R)
    assert np.abs(base - rotated).max() > 0.02


def test_flag_top_counts_and_ties():
    s = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    assert isf.flag_top(s, 0.4).tolist() == [False, True, False, True, False]
    assert isf.flag_top(s, 0.0).sum() == 1 and isf.flag_top(s, 1.0).sum() == 5


def test_tree_segments_cover_the_splits_inside_the_bounds():
    rng = np.random.default_rng(4)
    P = rng.standard_normal((40, 2))
    tree = isf.build_tree(P, rng, isf.height_limit(40))
    bounds = (P[:, 0].min() - 1, P[:, 0].max() + 1, P[:, 1].min() - 1, P[:, 1].max() + 1)
    segs = isf.tree_segments(tree, bounds)
    assert len(segs) == int((tree.feature >= 0).sum()) and segs[0][4] == 0
    for x0, y0, x1, y1, d in segs:
        assert bounds[0] - 1e-9 <= min(x0, x1) and max(x0, x1) <= bounds[1] + 1e-9 and bounds[2] - 1e-9 <= min(y0, y1) and max(y0, y1) <= bounds[3] + 1e-9
        assert x0 == x1 or y0 == y1                                                          # achsenparallel


def test_point_path_of_an_isolated_point_is_short():
    X = np.concatenate([np.random.default_rng(0).standard_normal((63, 2)), [[30.0, 30.0]]])
    depths = []
    for seed in range(20):
        tree = isf.build_tree(X, np.random.default_rng(seed), 6)
        depths.append((len(isf.point_path(tree, X[-1])[0]), np.mean([len(isf.point_path(tree, x)[0]) for x in X[:20]])))
    assert np.mean([d[0] for d in depths]) < 0.5 * np.mean([d[1] for d in depths])


# --- Kopierte Schätzer der Wurzel ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dof", [1, 2, 5, 12, 30])
@pytest.mark.parametrize("q", [0.5, 0.975, 0.999])
def test_chi2_matches_scipy(dof, q):
    assert alg.chi2_ppf(q, dof) == pytest.approx(chi2.ppf(q, dof), rel=1e-7)


def test_mcd_close_to_scikit_learn_and_affine_equivariant():
    X, is_out = _data(n=300, p=4, n_out=45, shift=8.0, seed=2)
    ours = alg.fit_mcd(X, 0.5, 0, reweight=True)
    sk = MinCovDet(random_state=0).fit(X)
    scale = np.sqrt(np.diag(np.cov(X[~is_out].T)))
    assert np.linalg.norm((ours.location - sk.location_) / scale) < 0.1
    assert (ours.d2[is_out] > alg.threshold(ours, 0.975)).all()
    A = np.random.default_rng(5).standard_normal((4, 4))
    assert np.allclose(alg.fit_mcd(X @ A.T + 2.0, 0.5, 0).d2, ours.d2, rtol=1e-5, atol=1e-6)
