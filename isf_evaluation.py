"""Auswertung der Isolation-Forest-Demo: Kennzahlen der Anomalie-Erkennung (AUC, mittlere Präzision, Precision/Recall/F1, Fehlalarmrate; aus der Wurzel-Demo übernommen), Analyse einer Aufnahme für den
Isolation Forest und die beiden Schätzer der Wurzel (klassisch, robust), Sweeps, Experimente auf Abruf, Score-Anisotropie und Urteil."""

import time
from dataclasses import dataclass

import numpy as np

import isf_algorithm as isf
import isf_ee_algorithm as alg
import isf_constants as C
import isf_scenario as sc


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------


def roc_auc(score, positive):
    """Fläche unter der ROC-Kurve über die Rangsumme (Mann-Whitney), Bindungen zählen halb. NaN, wenn eine Klasse fehlt."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = np.asarray(score)[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(score, positive):
    """Mittlere Präzision (Fläche unter der Precision-Recall-Kurve als Summe über die Treffer). NaN ohne Anomalien."""
    positive = np.asarray(positive, dtype=bool)
    if not positive.any():
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    precision_at = np.cumsum(hits) / (np.arange(len(hits)) + 1.0)
    return float(precision_at[hits].sum() / positive.sum())


def flag_metrics(flagged, positive):
    """Precision, Recall, F1 und Fehlalarmrate (Anteil der Normalen, die markiert werden). Ohne Anomalien: Recall/F1 NaN; ohne Markierung: Precision 1 (nichts falsch)."""
    flagged, positive = np.asarray(flagged, bool), np.asarray(positive, bool)
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if n_pos == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if n_pos and (precision + recall) > 0 else (float("nan") if not n_pos else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm": fp / n_neg if n_neg else float("nan"), "n_flagged": int(flagged.sum())}


def roc_curve(score, positive):
    """ROC-Kurve: (Fehlalarmrate, Trefferquote) für alle Schwellen, von (0, 0) bis (1, 1)."""
    positive = np.asarray(positive, bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    tpr = np.concatenate([[0.0], np.cumsum(hits) / max(hits.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~hits) / max((~hits).sum(), 1)])
    return fpr, tpr


# --- Analyse einer Aufnahme ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    n_trees: int = C.DEFAULT_TREES
    psi: int = C.DEFAULT_PSI
    threshold_kind: str = C.DEFAULT_THRESHOLD_KIND      # "standard": Score-Schwelle (Isolation Forest) und chi²-Quantil (klassisch, robust); "share": der erwartete Anteil für alle drei
    cutoff: float = C.DEFAULT_CUTOFF                    # Score-Schwelle des Isolation Forest (nominell 0.5)
    quantile: float = C.DEFAULT_QUANTILE                # chi²-Quantil der Schätzer der Wurzel
    share: int = C.DEFAULT_SHARE                        # erwarteter Anteil der Anomalien [%]
    start: int = 0                                      # Seed des Waldes und der MCD-Starts (entkoppelt vom Seed der Aufnahme)


DATA_KEYS = ("n", "p", "n_noise", "n_modes", "curvature", "noise", "contamination", "kind", "strength")
DEFAULT_DATA = dict(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                    contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH)
DETECTORS = ("iforest", "classical", "robust")
DETECTOR_NAMES = {"iforest": "Isolation Forest", "classical": "klassisch", "robust": "robust (MCD)"}
METRICS = ("auc", "ap", "precision", "recall", "f1", "false_alarm")


def make_dataset(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                 contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH, seed=C.DEFAULT_SEED):
    return sc.generate_dataset(n, p, n_modes, curvature, noise, contamination, kind, strength, seed, n_noise)


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    params: tuple                 # (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)
    forest: isf.Forest
    paths: np.ndarray             # (Bäume, n) Pfadlängen aller Touren
    values: dict                  # Detektor -> Anomalie-Wert je Tour (Isolation Forest: Score, klassisch/robust: quadrierter Mahalanobis-Abstand)
    classical: alg.Fit
    robust: alg.Fit
    scores: dict                  # Detektor -> Kennzahlen (auc, ap, precision, recall, f1, false_alarm, n_flagged, threshold)
    flags: dict                   # Detektor -> markierte Touren bei der gewählten Schwelle
    oracle_f1: float              # F1 des Isolation Forest, wenn der wahre Anteil bekannt wäre (die k größten Scores)
    seconds: dict


def _thresholds(a_values, ds, settings, classical, robust):
    """Markierung und Schwellenwert je Detektor: Standard = Score-Schwelle bzw. chi²-Quantil, sonst die k größten Werte mit dem angenommenen Anteil."""
    flags, thr = {}, {}
    if settings.threshold_kind == "share":
        for d in DETECTORS:
            flags[d] = isf.flag_top(a_values[d], settings.share / 100.0)
            thr[d] = float(np.sort(a_values[d])[::-1][int(flags[d].sum()) - 1])
    else:
        thr["iforest"] = settings.cutoff
        thr["classical"] = alg.threshold(classical, settings.quantile)
        thr["robust"] = alg.threshold(robust, settings.quantile)
        for d in DETECTORS:
            flags[d] = a_values[d] > thr[d]
    return flags, thr


def analyse(ds, settings=Settings(), params=None):
    secs = {}
    t0 = time.perf_counter()
    forest = isf.fit_forest(ds.X, settings.n_trees, settings.psi, settings.start)
    paths = isf.path_lengths(forest, ds.X)
    secs["iforest"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    classical = alg.fit_classical(ds.X)
    secs["classical"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    robust = alg.fit_mcd(ds.X, C.DEFAULT_SUPPORT, settings.start, reweight=C.DEFAULT_REWEIGHT)
    secs["robust"] = time.perf_counter() - t0
    values = {"iforest": isf.score_from_paths(paths, forest.psi), "classical": classical.d2, "robust": robust.d2}
    flags, thr = _thresholds(values, ds, settings, classical, robust)
    scores = {}
    for d in DETECTORS:
        m = flag_metrics(flags[d], ds.anomaly)
        m.update(auc=roc_auc(values[d], ds.anomaly), ap=average_precision(values[d], ds.anomaly), threshold=thr[d])
        scores[d] = m
    oracle = flag_metrics(isf.flag_top(values["iforest"], ds.anomaly.mean()), ds.anomaly)["f1"]
    return Analysis(ds, settings, params, forest, paths, values, classical, robust, scores, flags, oracle, secs)


def analyse_for(params, settings=Settings()):
    """`params` = (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)."""
    return analyse(make_dataset(*params), settings, params)


# --- Score-Anisotropie (Geister-Regionen achsenparalleler Schnitte) ------------------------------------------------------------------------


def anisotropy(X, forest, distance=1.0, grid=60):
    """Ein Isolation Forest über zwei Merkmale (Spalten von X) bewertet ein Raster um die Daten. Verglichen werden Rasterpunkte im GLEICHEN Abstand (standardisiert) zum nächsten Datenpunkt: solche, die
    außerhalb des Wertebereichs beider Merkmale liegen ("Ecken"), gegen solche, die in mindestens einem Bereich liegen ("Korridore"). Ein isotroper Detektor gäbe beiden dieselbe Bewertung;
    Rückgabe: mittlerer Score der Ecken minus der Korridore (NaN, wenn eine Gruppe zu klein ist) - positiv heißt: Punkte neben dem Datenbereich entlang der Achsen gelten als normaler (Geister-Regionen)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / sd
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    pad = 2.0 * distance
    xs = np.linspace(lo[0] - pad, hi[0] + pad, grid)
    ys = np.linspace(lo[1] - pad, hi[1] + pad, grid)
    G = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    dist = np.sqrt(((G[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
    s = isf.score(forest, G * sd + mu)
    band = np.abs(dist - distance) < 0.15
    corridor = ((G[:, 0] >= lo[0]) & (G[:, 0] <= hi[0])) | ((G[:, 1] >= lo[1]) & (G[:, 1] <= hi[1]))
    corner, side = s[band & ~corridor], s[band & corridor]
    if len(corner) < 4 or len(side) < 4:
        return float("nan")
    return float(corner.mean() - side.mean())


def score_map(X2, forest, grid=80, pad=0.5):
    """Scores eines Rasters über die zwei Merkmale (Wald nur auf diesen): (x-Achse, y-Achse, Scores [y, x])."""
    lo, hi = X2.min(axis=0), X2.max(axis=0)
    span = hi - lo
    xs = np.linspace(lo[0] - pad * span[0], hi[0] + pad * span[0], grid)
    ys = np.linspace(lo[1] - pad * span[1], hi[1] + pad * span[1], grid)
    G = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return xs, ys, isf.score(forest, G).reshape(grid, grid)


def ghost_probe(base_params, settings=Settings(), seed=C.DEFAULT_SEED, distance=1.0):
    """Die zwei ersten Merkmale (Distanz, Zeitfenster-Enge) ohne Anomalien: Anisotropie (Ecken gegen Korridore) und die Daten für die Score-Karte."""
    ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base_params, "n_noise": 0, "p": 2})
    X2 = ds.X[~ds.anomaly]
    forest = isf.fit_forest(X2, settings.n_trees, settings.psi, settings.start)
    return X2, forest, anisotropy(X2, forest, distance)


# --- Sweeps und Experimente -----------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n": (20, 30, 50, 100, 200, 400, 600),
    "p": (2, 5, 8, 12, 20, 30),
    "n_noise": (0, 5, 10, 20, 30, 40),
    "n_modes": (1, 2, 3),
    "curvature": (0.0, 0.25, 0.5, 0.75, 1.0),
    "noise": (0.0, 0.25, 0.5, 0.75, 1.0),
    "contamination": (2, 5, 10, 20, 30, 40, 45),
    "strength": (3.0, 4.0, 6.0, 9.0, 12.0),
    "n_trees": (10, 25, 50, 100, 200, 500),
    "psi": (16, 32, 64, 128, 256),
    "cutoff": (0.45, 0.5, 0.55, 0.6, 0.65),
    "quantile": (0.9, 0.95, 0.975, 0.99, 0.999),
    "share": (2, 5, 10, 20, 40),
}
SWEEP_LABELS = {"n": "Anzahl Touren", "p": "Anzahl Merkmale", "n_noise": "Anzahl Rauschmerkmale", "n_modes": "Anzahl Betriebsarten", "curvature": "Krümmung des Normalbereichs", "noise": "Rauschen",
                "contamination": "Anteil der Anomalien [%]", "strength": "Abstand der Anomalien (Faktor-σ)", "n_trees": "Anzahl Bäume", "psi": "Größe der Unterstichprobe ψ",
                "cutoff": "Score-Schwelle des Isolation Forest", "quantile": "chi²-Quantil der Schwelle (klassisch, robust)", "share": "angenommener Anteil der Anomalien [%]"}
SETTING_PARAMETERS = ("n_trees", "psi", "cutoff", "quantile", "share")


def _record(a):
    out = {f"{d}_{k}": a.scores[d][k] for d in DETECTORS for k in METRICS}
    out["n_anomalies"] = float(a.ds.anomaly.sum())
    out["iforest_oracle_f1"] = a.oracle_f1
    return out


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        ok = not np.isnan(arr).all()
        row[key] = float(np.nanmean(arr)) if ok else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if ok else float("nan")
        row[key + "_min"] = float(np.nanmin(arr)) if ok else float("nan")
        row[key + "_max"] = float(np.nanmax(arr)) if ok else float("nan")
    return row


def _mean_over_seeds(settings=Settings(), seeds=C.SWEEP_SEEDS, **kw):
    """Mittel (mit Streuung und Spanne) aller Kennzahlen über die festen Sweep-Datensätze für eine Datenkonfiguration."""
    return _summarise(None, [_record(analyse(make_dataset(seed=s, **{**DEFAULT_DATA, **kw}), settings)) for s in seeds])


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel, Streuung und Spanne der Kennzahlen der drei Detektoren über die festen Sweep-Datensätze in Abhängigkeit von einem Regler (alle anderen wie in `base`)."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        if parameter in SETTING_PARAMETERS:
            row = _mean_over_seeds(Settings(**{**settings.__dict__, parameter: x}), **base)
        else:
            row = _mean_over_seeds(settings, **{**base, parameter: x})
        row["x"] = x
        rows.append(row)
    return rows


SPREAD_TREES = (10, 50, 200)
SPREAD_STARTS = 6


def parameter_table(settings=Settings(), **base):
    """Wald-Parameter: AUC und F1 über Bäume und Unterstichprobe (Mittel über die Sweep-Datensätze) und die Streuung über sechs Wald-Seeds bei 10, 50 und 200 Bäumen."""
    trees = sweep("n_trees", settings=settings, **base)
    psis = sweep("psi", settings=settings, **base)
    spread = []
    for n_trees in SPREAD_TREES:
        aucs, f1s = [], []
        for seed in C.SWEEP_SEEDS:
            ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base})
            au, ff = [], []
            for start in range(SPREAD_STARTS):
                values = isf.score(isf.fit_forest(ds.X, n_trees, settings.psi, start), ds.X)
                flags = isf.flag_top(values, settings.share / 100.0) if settings.threshold_kind == "share" else values > settings.cutoff
                au.append(roc_auc(values, ds.anomaly))
                ff.append(flag_metrics(flags, ds.anomaly)["f1"])
            aucs.append(np.std(au))
            f1s.append(np.std(ff))
        spread.append({"n_trees": n_trees, "auc_std": float(np.mean(aucs)), "f1_std": float(np.mean(f1s))})
    return {"trees": trees, "psi": psis, "spread": spread}


def modes_table(settings=Settings(), **base):
    """Betriebsarten × Art der Anomalien: AUC der drei Detektoren (die Art 'Lücke' gibt es erst ab zwei Betriebsarten)."""
    rows = []
    for n_modes in (1, 2, 3):
        for kind in C.KINDS:
            if kind == "gap" and n_modes == 1:
                continue
            row = _mean_over_seeds(settings, **{**base, "n_modes": n_modes, "kind": kind})
            row.update(n_modes=n_modes, kind=kind)
            rows.append(row)
    return rows


MASKING_CONTAMINATION = (2, 5, 10, 15, 20, 25, 30, 35, 40, 45)


def masking_table(settings=Settings(), **base):
    """Dichte Gruppe abseits: AUC der drei Detektoren über den Anteil, dazu der Einfluss der Unterstichprobe ψ bei 30 % (kleine Unterstichprobe gegen Masking)."""
    rows = []
    for c in MASKING_CONTAMINATION:
        row = _mean_over_seeds(settings, **{**base, "kind": "cluster", "contamination": c, "n_modes": 1})
        row["x"] = c
        rows.append(row)
    psi_rows = []
    for psi in (16, 32, 64, 128, 256):
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "psi": psi}), **{**base, "kind": "cluster", "contamination": 30, "n_modes": 1})
        row["x"] = psi
        psi_rows.append(row)
    return {"rows": rows, "psi": psi_rows}


def threshold_table(settings=Settings(), **base):
    """Schwelle: (1) Kennzahlen des Isolation Forest über die Score-Schwelle; (2) F1 aller drei Detektoren, wenn der angenommene Anteil das ½-, 1- und 2-fache des wahren ist;
    (3) der mittlere Score der normalen Touren je Tourenzahl (die nominelle Schwelle 0.5 ist nur ungefähr)."""
    cut = sweep("cutoff", settings=Settings(**{**settings.__dict__, "threshold_kind": "standard"}), **base)
    true_share = base.get("contamination", C.DEFAULT_CONTAMINATION)
    wrong = []
    for factor in (0.5, 1.0, 2.0):
        share = int(round(true_share * factor))
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "threshold_kind": "share", "share": share}), **base)
        row["x"], row["factor"] = share, factor
        wrong.append(row)
    normal_scores = []
    for n in (20, 50, 100, 300, 600):
        meds = []
        for seed in C.SWEEP_SEEDS:
            a = analyse(make_dataset(seed=seed, **{**DEFAULT_DATA, **base, "n": n}), settings)
            meds.append(np.median(a.values["iforest"][~a.ds.anomaly]))
        normal_scores.append({"n": n, "median": float(np.mean(meds))})
    return {"cutoff": cut, "wrong_share": wrong, "normal_scores": normal_scores}


DIMENSION_N = (20, 30, 50, 100, 200, 400)
DIMENSION_P = (2, 5, 12, 20, 30)


def dimension_table(settings=Settings(), **base):
    """Hohe Dimension: AUC des Isolation Forest und der robusten Schätzung über Tourenzahl n (Zeilen) und Merkmalszahl p (Spalten)."""
    cells = []
    for n in DIMENSION_N:
        for p in DIMENSION_P:
            row = _mean_over_seeds(settings, **{**base, "n": n, "p": p})
            row.update(n=n, p=p)
            cells.append(row)
    return cells


def ghost_table(settings=Settings(), **base):
    """Score-Anisotropie über Betriebsarten (1-3) und Abstand zum Datenrand (0.5, 1, 2 Streuungen): Mittel über die Sweep-Datensätze."""
    rows = []
    for n_modes in (1, 2, 3):
        for distance in (0.5, 1.0, 2.0):
            vals = [ghost_probe({**base, "n_modes": n_modes}, settings, seed, distance)[2] for seed in C.SWEEP_SEEDS]
            rows.append({"n_modes": n_modes, "distance": distance, "mean": float(np.nanmean(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)), "n_valid": int(np.sum(~np.isnan(vals)))})
    return rows


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.05             # AUC-Abstand, ab dem ein Sieger benannt wird
GAP_AUC = 0.8
THRESHOLD_F1_DROP = 0.15      # F1-Verlust gegen die Schwelle mit bekanntem Anteil


def verdict(a):
    """(Art, Code, Kennzahlen): Lücke, dichte Gruppe (Wurzel besser), Wurzel besser, Isolation Forest besser (die Meldung nennt eine schlechte Schwelle mit), falsche Schwelle bei guter Rangfolge, sonst gleichauf."""
    ds, s = a.ds, a.settings
    i, c, r = a.scores["iforest"], a.scores["classical"], a.scores["robust"]
    best_root = max(c["auc"], r["auc"])
    data = {"n": ds.n, "p": ds.p + ds.n_noise, "n_noise": ds.n_noise, "n_modes": ds.n_modes, "kind": ds.kind, "contamination": 100.0 * ds.anomaly.mean(), "n_anomalies": int(ds.anomaly.sum()),
            "best_root_auc": best_root, "oracle_f1": a.oracle_f1, **{f"{d}_{k}": v for d in DETECTORS for k, v in a.scores[d].items()}}
    if ds.kind == "gap" and i["auc"] < GAP_AUC:
        return "warning", "gap", data
    if r["auc"] - i["auc"] >= WIN_MARGIN:
        return "warning", "dense_masking" if ds.kind == "cluster" else "root_wins", data
    if i["auc"] - best_root >= WIN_MARGIN:
        return "success", "if_wins", data
    if i["auc"] >= 0.95 and a.oracle_f1 - i["f1"] > THRESHOLD_F1_DROP:
        return "warning", "threshold_off", data
    return "success", "comparable", data
