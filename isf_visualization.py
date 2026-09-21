"""Plotly-Visualisierungen der Isolation-Forest-Demo: Touren in der Ebene der größten Streuung, ein Isolationsbaum als Zerlegung der Ebene, Pfadlängen, Scores und Schwelle, ROC-Kurven, Kennzahlen-Balken,
Sweeps, Wald-Parameter, Score-Karte (Geister-Regionen), Betriebsarten, Masking, Schwelle und Dimension. Alle Figuren laufen durch `lock_axes`."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import isf_ee_algorithm as alg

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f"
DETECTOR_COLORS = {"iforest": PURPLE, "classical": ORANGE, "robust": TEAL}
DETECTOR_NAMES = {"iforest": "Isolation Forest", "classical": "klassisch", "robust": "robust (MCD)"}
KIND_NAMES = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke"}
DETECTORS = ("iforest", "classical", "robust")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# --- Projektion -----------------------------------------------------------------------------------------------------------------


def standardise(X):
    m, s = X.mean(axis=0), X.std(axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    return (X - m) / s


def projection(X, robust):
    """Die Touren in der Ebene der zwei größten Streuungsrichtungen der robusten Kovarianz (standardisierte Kennzahlen): (Punkte n x 2, Achsen)."""
    s = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
    axes = alg.projection_axes(robust.covariance / np.outer(s, s))
    return standardise(X) @ axes, axes


def _points(P, anomaly, flagged=None):
    normal = ~anomaly
    traces = [go.Scatter(x=P[normal, 0], y=P[normal, 1], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55), hoverinfo="skip", name="normale Touren"),
              go.Scatter(x=P[anomaly, 0], y=P[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond"), hoverinfo="skip", name="Sonderfahrten (Wahrheit)")]
    if flagged is not None and flagged.any():
        traces.append(go.Scatter(x=P[flagged, 0], y=P[flagged, 1], mode="markers", marker=dict(size=13, color="black", line=dict(width=2), symbol="circle-open"), hoverinfo="skip", name="als Anomalie markiert"))
    return traces


def build_scatter(P, anomaly, flagged=None, height=380):
    """Touren in der Projektionsebene (blau = normal, rote Rauten = Sonderfahrten, Kreise = als Anomalie markiert)."""
    fig = go.Figure(_points(P, anomaly, flagged))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1 (standardisiert)", zeroline=False), yaxis=dict(title="Hauptrichtung 2", zeroline=False),
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_features(X, anomaly, names, i=0, j=1):
    """Zwei Rohmerkmale gegeneinander (Einheiten wie gemessen)."""
    j = min(j, X.shape[1] - 1)
    fig = go.Figure(_points(np.stack([X[:, i], X[:, j]], axis=1), anomaly))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=names[i], zeroline=False), yaxis=dict(title=names[j], zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_tree_plane(P, anomaly, segments, picks, n_show=None):
    """Ein Isolationsbaum, auf den zwei Hauptrichtungen der Ebene gebaut (Illustration desselben Algorithmus): die Schnittlinien (dunkler = früher), dazu ein normaler Punkt (Kreis) und eine Anomalie (Raute),
    deren Weg durch den Baum im Text daneben steht. `picks` = [(Index, Name, Farbe)]."""
    fig = go.Figure(_points(P, anomaly))
    if segments:
        max_depth = max(s[4] for s in segments)
        for x0, y0, x1, y1, d in segments:
            shade = 0.25 + 0.75 * (1.0 - d / max(max_depth, 1))
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=f"rgba(60,60,60,{shade:.2f})", width=1.6 if d < 3 else 1), hoverinfo="skip", showlegend=False))
    for idx, name, color in picks:
        fig.add_trace(go.Scatter(x=[P[idx, 0]], y=[P[idx, 1]], mode="markers", marker=dict(size=16, color=color, line=dict(width=3), symbol="circle-open"), name=name, hoverinfo="skip"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1", zeroline=False, range=[P[:, 0].min() - 0.5, P[:, 0].max() + 0.5]),
                      yaxis=dict(title="Hauptrichtung 2", zeroline=False, range=[P[:, 1].min() - 0.5, P[:, 1].max() + 0.5]), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_path_lengths(paths_normal, paths_anomaly, c_psi):
    """Pfadlänge eines normalen und eines Sonderfahrt-Punkts über alle Bäume des Waldes (Histogramm); senkrecht: c(ψ), die mittlere Pfadlänge einer erfolglosen Suche."""
    fig = go.Figure()
    top = max(float(paths_normal.max()), float(paths_anomaly.max()), c_psi) + 1
    bins = dict(start=0.0, end=top, size=top / 30.0)
    fig.add_trace(go.Histogram(x=paths_normal, xbins=bins, marker_color=BLUE, opacity=0.7, name="normaler Punkt", hoverinfo="skip"))
    fig.add_trace(go.Histogram(x=paths_anomaly, xbins=bins, marker_color=RED, opacity=0.8, name="Sonderfahrt", hoverinfo="skip"))
    fig.add_vline(x=float(c_psi), line=dict(color="black", dash="dash"), annotation_text="c(ψ)", annotation_position="top")
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title="Pfadlänge in einem Baum"), yaxis=dict(title="Bäume"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_mean_paths(mean_paths, anomaly, c_psi):
    """Mittlere Pfadlänge je Tour über alle Bäume (Histogramm): die Sonderfahrten sind schneller isoliert."""
    top = float(max(mean_paths.max(), c_psi)) + 1
    bins = dict(start=0.0, end=top, size=top / 40.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mean_paths[~anomaly], xbins=bins, marker_color=BLUE, opacity=0.7, name="normale Touren", hoverinfo="skip"))
    fig.add_trace(go.Histogram(x=mean_paths[anomaly], xbins=bins, marker_color=RED, opacity=0.8, name="Sonderfahrten", hoverinfo="skip"))
    fig.add_vline(x=float(c_psi), line=dict(color="black", dash="dash"), annotation_text="c(ψ)", annotation_position="top")
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title="mittlere Pfadlänge E[h]"), yaxis=dict(title="Touren"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_convergence(steps, curves, anomaly_flags):
    """Anomalie-Wert einiger Touren nach den ersten k Bäumen: er wird mit mehr Bäumen stabil. `curves` (len(steps), m), `anomaly_flags` (m,)."""
    fig = go.Figure()
    for j in range(curves.shape[1]):
        fig.add_trace(go.Scatter(x=[str(k) for k in steps], y=curves[:, j], mode="lines", line=dict(color=RED if anomaly_flags[j] else BLUE, width=2), opacity=0.8, hoverinfo="skip", showlegend=False))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Anzahl Bäume", type="category"), yaxis=dict(title="Anomalie-Wert", range=[0.2, 1.0]))
    return lock_axes(fig)


def build_scores(values, anomaly, thr, name, higher_is_anomaly=True):
    """Histogramm des Anomalie-Werts (Isolation Forest: Score) der normalen und der Sonderfahrten; senkrechte Linie = Schwelle."""
    top = float(max(values.max(), thr * 1.05))
    lo = float(min(values.min(), thr * 0.95))
    bins = dict(start=lo, end=top, size=(top - lo) / 40.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values[~anomaly], xbins=bins, marker_color=BLUE, opacity=0.7, name="normale Touren", hoverinfo="skip"))
    if anomaly.any():
        fig.add_trace(go.Histogram(x=values[anomaly], xbins=bins, marker_color=RED, opacity=0.8, name="Sonderfahrten", hoverinfo="skip"))
    fig.add_vline(x=float(thr), line=dict(color="black", dash="dash"), annotation_text="Schwelle", annotation_position="top")
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title=name), yaxis=dict(title="Touren"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_roc(curves):
    """ROC-Kurven (Fehlalarmrate gegen Trefferquote) der drei Detektoren; `curves` = {Detektor: (fpr, tpr, auc)}."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=DETECTOR_COLORS[name], width=3), name=f"{DETECTOR_NAMES[name]} (AUC {auc:.2f})", hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Fehlalarmrate", range=[0, 1]), yaxis=dict(title="Trefferquote", range=[0, 1.02]), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_method_bars(scores):
    """Kennzahlen der drei Detektoren nebeneinander: AUC, Recall, Precision, Fehlalarmrate (bei der gewählten Schwelle)."""
    keys = (("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision"), ("false_alarm", "Fehlalarmrate"))
    fig = go.Figure()
    for det in DETECTORS:
        y = [scores[det][k] for k, _ in keys]
        fig.add_trace(go.Bar(x=[lab for _, lab in keys], y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=10), hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


# --- Sweeps und Experimente ----------------------------------------------------------------------------------------------------------


def _band(fig, xs, rows, key, color, col):
    y, sd = np.array([r[key] for r in rows]), np.array([r[key + "_std"] for r in rows])
    fig.add_trace(go.Scatter(x=list(xs) + list(xs)[::-1], y=list(np.nan_to_num(y + sd)) + list(np.nan_to_num(y - sd))[::-1], fill="toself", fillcolor=color, opacity=0.13, line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=col)


def build_sweep(rows, xlabel, current=None):
    """Links AUC der drei Detektoren (mit Streuung über die Sweep-Datensätze), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC (Rangfolge)", "F1 und Fehlalarmrate bei der Schwelle"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    for det in DETECTORS:
        color = DETECTOR_COLORS[det]
        _band(fig, xs, rows, f"{det}_auc", color, 1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=color, width=3), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_f1"] for r in rows], mode="lines+markers", line=dict(color=color, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_false_alarm"] for r in rows], mode="lines+markers", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_parameters(trees, psis):
    """Wald-Parameter: AUC (durchgezogen) und F1 bei der Schwelle (gestrichelt) des Isolation Forest über die Anzahl Bäume (links) und die Unterstichprobe ψ (rechts, Mittel mit Streuung)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Anzahl Bäume", "Größe der Unterstichprobe ψ"), horizontal_spacing=0.12)
    for col, rows in ((1, trees), (2, psis)):
        xs = [r["x"] for r in rows]
        _band(fig, xs, rows, "iforest_auc", PURPLE, col)
        _band(fig, xs, rows, "iforest_f1", ORANGE, col)
        fig.add_trace(go.Scatter(x=xs, y=[r["iforest_auc"] for r in rows], mode="lines+markers", name="AUC", line=dict(color=PURPLE, width=3), hoverinfo="skip", showlegend=col == 1), row=1, col=col)
        fig.add_trace(go.Scatter(x=xs, y=[r["iforest_f1"] for r in rows], mode="lines+markers", name="F1 bei der Schwelle", line=dict(color=ORANGE, width=3, dash="dash"), hoverinfo="skip", showlegend=col == 1), row=1, col=col)
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_ghost_map(xs, ys, S, X2, names, ellipse=None):
    """Score-Karte eines Isolation Forest über zwei Merkmale (dunkler = normaler, heller = auffälliger) mit den Touren; die Bänder entlang der Achsen sind die Geister-Regionen der achsenparallelen Schnitte.
    Optional die Ellipse der robusten Schätzung derselben zwei Merkmale (gestrichelt)."""
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=S, colorscale="Viridis", reversescale=True, colorbar=dict(title="Score", thickness=12), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=X2[:, 0], y=X2[:, 1], mode="markers", marker=dict(size=4, color="white", line=dict(color="black", width=0.5)), hoverinfo="skip", name="Touren"))
    if ellipse is not None:
        fig.add_trace(go.Scatter(x=ellipse[:, 0], y=ellipse[:, 1], mode="lines", line=dict(color=TEAL, width=3, dash="dash"), hoverinfo="skip", name="robuste Ellipse"))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=names[0], zeroline=False), yaxis=dict(title=names[1], zeroline=False), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_anisotropy(rows):
    """Score-Anisotropie (Ecken minus Korridore bei gleichem Abstand zum Datenrand) je Betriebsart und Abstand: positiv = Geister-Regionen entlang der Achsen."""
    fig = go.Figure()
    for distance, color in ((1.0, PURPLE), (2.0, ORANGE)):
        sel = [r for r in rows if r["distance"] == distance and r["n_valid"] >= 3]
        fig.add_trace(go.Bar(x=[f"{r['n_modes']} Betriebsart{'en' if r['n_modes'] > 1 else ''}" for r in sel], y=[r["mean"] for r in sel], name=f"Abstand {distance:g} σ", marker_color=color,
                             text=[f"{r['mean']:.3f}" for r in sel], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Ecken − Korridore", range=[0, 0.14]), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_modes(rows):
    """AUC der drei Detektoren je Anzahl Betriebsarten und Art der Anomalien (die Linie bei 0.5 ist Raten)."""
    labels = [f"{r['n_modes']} Betriebsart{'en' if r['n_modes'] > 1 else ''}<br>{KIND_NAMES[r['kind']]}" for r in rows]
    fig = go.Figure()
    for det in DETECTORS:
        y = [r[f"{det}_auc"] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=8), hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=400, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="AUC", range=[0, 1.15]), legend=dict(orientation="h", y=-0.5))
    return lock_axes(fig)


def build_masking(table):
    """Dichte Gruppe abseits: AUC der drei Detektoren über den Anteil (links) und der Isolation Forest bei 30 % über die Unterstichprobe ψ (rechts)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC über den Anteil der Gruppe", "Isolation Forest bei 30 %: Unterstichprobe ψ"), horizontal_spacing=0.12)
    xs = [r["x"] for r in table["rows"]]
    for det in DETECTORS:
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in table["rows"]], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=DETECTOR_COLORS[det], width=3), hoverinfo="skip"), row=1, col=1)
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"), row=1, col=1)
    ps = table["psi"]
    fig.add_trace(go.Scatter(x=[r["x"] for r in ps], y=[r["iforest_auc"] for r in ps], mode="lines+markers", line=dict(color=PURPLE, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.update_xaxes(title="Anteil der Gruppe [%]", row=1, col=1)
    fig.update_xaxes(title="ψ", row=1, col=2)
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_cutoff(rows):
    """Precision, Recall, Fehlalarmrate und F1 des Isolation Forest über die Score-Schwelle."""
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("iforest_precision", "Precision", GREEN, "solid"), ("iforest_recall", "Recall", TEAL, "solid"), ("iforest_false_alarm", "Fehlalarmrate", RED, "dash"), ("iforest_f1", "F1", PURPLE, "solid")):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=3 if key == "iforest_f1" else 2, dash=dash), hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Score-Schwelle", type="category"), yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_wrong_share(rows):
    """F1 der drei Detektoren, wenn der angenommene Anteil das ½-, 1- und 2-fache des wahren ist."""
    xs = [f"{r['x']} % ({r['factor']:g}×)" for r in rows]
    fig = go.Figure()
    for det in DETECTORS:
        y = [r[f"{det}_f1"] for r in rows]
        fig.add_trace(go.Bar(x=xs, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="angenommener Anteil (Vielfaches des wahren)"), yaxis=dict(title="F1", range=[0, 1.15]),
                      legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_dimension(cells):
    """AUC des Isolation Forest (links) und der robusten Schätzung (rechts) über Tourenzahl n (Zeilen) und Merkmalszahl p (Spalten)."""
    ns = sorted({c["n"] for c in cells})
    ps = sorted({c["p"] for c in cells})
    grid = {(c["n"], c["p"]): c for c in cells}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC Isolation Forest", "AUC robust (MCD)"), horizontal_spacing=0.14)
    for col, key, scale in ((1, "iforest_auc", "Purples"), (2, "robust_auc", "Teal")):
        z = [[grid[(n, p)][key] for p in ps] for n in ns]
        fig.add_trace(go.Heatmap(z=z, x=[f"p = {p}" for p in ps], y=[f"n = {n}" for n in ns], colorscale=scale, zmin=0.3, zmax=1, text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}", showscale=False,
                                 hoverinfo="skip"), row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)
