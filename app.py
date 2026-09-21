"""Isolation Forest - Anomalien, die sich schnell isolieren lassen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den Isolation Forest - und lässt stattdessen das Beispiel wachsen.
Zweites Stück der Anomalie-Erkennung-Linie der "Konzepte"-Reihe: ein unabhängiger Ast direkt nach der Wurzel (Elliptic Envelope), der deren Annahmen (ein Normalbereich, Gauß, viele Touren je Merkmal)
auf einem anderen Weg umgeht - mit eigenen Schwächen (achsenparallele Schnitte, Masking, nicht kalibrierte Schwelle). Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import isf_algorithm as isf
import isf_constants as C
import isf_ee_algorithm as alg
from isf_evaluation import (
    SWEEP_LABELS,
    Settings,
    analyse_for,
    dimension_table,
    ghost_probe,
    ghost_table,
    masking_table,
    modes_table,
    parameter_table,
    roc_curve,
    score_map,
    sweep,
    threshold_table,
    verdict,
)
from isf_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    kind_options,
    load_permalink_settings,
    psi_max,
    randomize_seed,
    sync_query_params,
)
from isf_visualization import (
    build_anisotropy,
    build_convergence,
    build_cutoff,
    build_dimension,
    build_features,
    build_ghost_map,
    build_masking,
    build_mean_paths,
    build_method_bars,
    build_modes,
    build_parameters,
    build_path_lengths,
    build_roc,
    build_scatter,
    build_scores,
    build_sweep,
    build_tree_plane,
    build_wrong_share,
    projection,
)

st.set_page_config(page_title="Isolation Forest – Sebastian Hanisch", layout="wide")
BLUE_TXT, RED_TXT = "#1f77b4", "#d62728"


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings):
    return sweep(parameter, settings=settings, **dict(base))


@st.cache_data(show_spinner=False)
def _parameters(base, settings):
    return parameter_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _ghost(base, settings, seed):
    X2, forest, aniso = ghost_probe(dict(base), settings, seed)
    xs, ys, S = score_map(X2, forest)
    robust = alg.fit_mcd(X2, C.DEFAULT_SUPPORT, 0)
    ellipse = alg.ellipse_points(robust.location, robust.covariance, alg.chi2_ppf(0.975, 2), np.eye(2))
    return X2, xs, ys, S, ellipse, aniso, ghost_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _modes(base, settings):
    return modes_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _dimension(base, settings):
    return dimension_table(settings, **dict(base)), sweep("n_noise", settings=settings, **dict(base))


@st.cache_data(show_spinner=False)
def _threshold(base, settings):
    return threshold_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _masking(base, settings):
    return masking_table(settings, **dict(base))


st.title("🌲 Isolation Forest – Anomalien sind leicht zu isolieren")
st.markdown(
    """
Die Wurzel der Linie, der **Elliptic Envelope**, beschreibt die normalen Touren als **eine Gauß'sche Wolke** und markiert, was außerhalb der Ellipse liegt. Der **Isolation Forest** dreht die Frage um: er beschreibt nichts Normales,
sondern fragt, **wie schnell sich eine Tour vom Rest abtrennen lässt**. Dazu werden zufällige Schnitte gemacht - ein zufälliges Merkmal, ein zufälliger Schnittpunkt zwischen kleinstem und größtem Wert -, immer weiter, bis jede Tour allein ist.
Eine Sonderfahrt liegt abseits und ist nach wenigen Schnitten allein (**kurzer Pfad**), eine normale Tour steckt mitten im Rest und braucht viele. Der Mittelwert der Pfadlängen über viele solcher Bäume ergibt den **Anomalie-Wert**.
Keine Dichte, keine Abstände, kein Formmodell - dafür eigene Schwächen. Was das gegenüber der Wurzel bringt und wo es endet, misst diese Demo - mit Siegen und Niederlagen.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - zweites Stück der Anomalie-Erkennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Szenario und die Schätzer der Wurzel (klassisch, robust per MCD) sind wortgleich aus der elliptic-envelope-demo übernommen, damit der Vergleich derselbe Boden hat. Die Linie hat keinen Konvergenzpunkt: die nächsten Stücke "
    "(Extended Isolation Forest, lokale Dichte, Kernel-Grenze, ...) beheben jeweils eine Schwäche eines dieser Wege auf einem anderen Weg. Die Zufallsbäume verwandt mit dem Bagging der Baum-Verfahren (dort noch nicht gebaut)."
)

with st.expander("So funktioniert der Isolation Forest", expanded=True):
    st.markdown(
        """
1. **Unterstichprobe.** Jeder Baum sieht nur $\\psi$ zufällig gezogene Touren (Standard 256), nicht alle. Das ist Absicht: kleine Stichproben verhindern, dass dichte Anomaliegruppen sich gegenseitig verdecken (Masking).
2. **Zufälliger Baum.** Je Knoten: ein zufälliges Merkmal und ein **gleichverteilter Schnittpunkt** zwischen Minimum und Maximum der Touren im Knoten. Links landet, was kleiner ist, rechts der Rest. Weiter, bis jede Tour allein ist oder die Tiefe $\\lceil \\log_2 \\psi \\rceil$ erreicht ist.
3. **Pfadlänge.** $h(x)$ = Zahl der Schnitte bis $x$ allein ist (plus $c(n)$ für die nicht mehr getrennten Touren im Blatt). Anomalien haben kurze Pfade.
4. **Anomalie-Wert.** $s(x) = 2^{-E[h(x)] / c(\\psi)}$, $c(\\psi) = 2H(\\psi-1) - 2(\\psi-1)/\\psi$: nahe 1 = sehr leicht zu isolieren, um 0,5 = normal (kein Unterschied zu einer erfolglosen Suche im Suchbaum), nahe 0 = sehr dicht.
5. **Schwelle.** Es gibt keine Verteilungsannahme, deshalb keine kalibrierte Fehlalarmrate: entweder die Faustregel "Score über 0,5" oder ein **erwarteter Anteil** (die größten Werte markieren) - beides misst diese Demo.

Was **nicht** vorausgesetzt wird: eine Verteilungsform, ein Normalbereich, mehr Touren als Merkmale. Was **nicht** gilt: Drehungsinvarianz - die Schnitte sind **achsenparallel**, gedrehte Daten werden anders bewertet (Ghost-Regionen).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider(
        "Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=10,
        help="Anzahl der Touren. Mit der Schwelle 0.5 sinkt die Fehlalarmrate des Isolation Forest 0.156 / 0.119 / 0.062 / 0.027 / 0.009 bei 20 / 30 / 50 / 100 / 200 Touren (F1 0.60 / 0.67 / 0.78 / 0.89 / 0.96): die normalen Scores liegen bei "
             "kleinen Stichproben höher. Die AUC ist überall 1.00. Zum Vergleich: der klassische Recall ist 0.00 / 0.00 / 0.20 / 0.42 / 0.44, die Fehlalarmrate der robusten Schätzung 0.16 / 0.24 / 0.29 / 0.21 / 0.06.",
    )
    p_features = st.slider(
        "Merkmale", *bounds("p_slider"), key="p_slider",
        help="Anzahl der Kennzahlen je Tour (ab 13 zusätzliche Mischungen der versteckten Faktoren). Der Isolation Forest bleibt bei 2 bis 30 Merkmalen bei AUC 1.00 und F1 0.92-0.96; der klassische Recall fällt von 0.84 auf 0.21, "
             "die Fehlalarmrate der robusten Schätzung steigt von 0.03 auf 0.16 (bei 300 Touren und 10 % Anomalien).",
    )
    n_noise = st.slider(
        "Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider", step=5,
        help="Zusätzliche unabhängige Spalten ohne Zusammenhang mit den Faktoren. Bei 0 / 10 / 20 / 30 / 40: AUC des Isolation Forest 1.00 / 1.00 / 1.00 / 0.99 / 0.99 (robust 1.00 / 0.99 / 0.96 / 0.91 / 0.88, klassisch 0.95 / 0.90 / 0.86 / 0.82 / 0.79) - "
             "aber sein Recall bei der Schwelle 0.5 fällt auf 0.99 / 0.91 / 0.71 / 0.58 / 0.39 (F1 0.96 / 0.94 / 0.82 / 0.72 / 0.56): viele Schnitte treffen irrelevante Merkmale, die Scores der Anomalien rücken an 0.5 heran.",
    )
    n_modes = st.slider(
        "Betriebsarten", *bounds("n_modes_slider"), key="n_modes_slider",
        help="Aus wie vielen Gruppen (Stadt, Land, Fernverkehr) die normalen Touren stammen. Der Isolation Forest hat bei 1 / 2 / 3 Betriebsarten Recall 0.99 / 0.99 / 0.97 (verstreute Anomalien), die robuste Schätzung der Wurzel 0.97 / 0.77 / 0.47 "
             "(AUC 1.00 / 0.95 / 0.85).",
    )
    curvature = st.slider(
        "Krümmung des Normalbereichs", *bounds("curvature_slider"), key="curvature_slider", step=0.25,
        help="Biegt die normale Fläche (nicht mehr konvex). Bei 0 / 0.25 / 0.5 / 0.75 / 1 bleibt der F1 des Isolation Forest bei 0.96 / 0.98 / 0.97 / 0.97 / 0.95; die robuste Schätzung mit χ²-Schwelle fällt auf 0.84 / 0.49 / 0.41 / 0.39 / 0.38 "
             "(Fehlalarmrate 0.04 auf 0.36), die klassische steigt auf 0.90-0.91. Die Rangfolge (AUC 1.00) ist für alle drei gleich - der Unterschied liegt in der Schwelle.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen der Kennzahlen (relativ zu den versteckten Faktoren). Bei 0 / 0.25 / 0.5 / 1.0: Recall des Isolation Forest 0.99 / 0.99 / 0.97 / 0.94 (AUC 1.00), klassisch 0.87 / 0.43 / 0.42 / 0.38.",
    )
    contamination = st.slider(
        "Anteil der Anomalien [%]", *bounds("contamination_slider"), key="contamination_slider",
        help="Wie viele Touren Sonderfahrten sind (mindestens eine). Bei verstreuten Ausreißern hat der Isolation Forest F1 0.42 / 0.81 / 0.96 / 0.97 / 0.93 / 0.87 / 0.81 bei 2 / 5 / 10 / 20 / 30 / 40 / 45 % (AUC immer 1.00; "
             "die Schwelle 0.5 markiert bei sehr wenigen Anomalien viele Normale); robust 0.46 / 0.71 / 0.84 / 0.92 / 0.93 / 0.85 / 0.53, klassisch Recall 0.93 / 0.75 / 0.43 / 0.14 / 0.06 / 0.04 / 0.04.",
    )
    kind = st.selectbox(
        "Art der Anomalien", kind_options(n_modes), key="kind_select", format_func=lambda k: C.KIND_LABELS[k],
        help="Verstreut: jede Anomalie in einer anderen Richtung. Dichte Gruppe: alle beieinander - bei 10 % hat der Isolation Forest AUC 0.95, die robuste Schätzung 1.00, die klassische 0.83; bei 30 % IF 0.70, robust 0.49. "
             "In der Lücke (ab zwei Betriebsarten): AUC des Isolation Forest 0.54, klassisch und robust 0.40 - kaum besser als Raten.",
    )
    if kind != "gap":
        strength = st.slider(
            "Abstand der Anomalien (Faktor-σ)", *bounds("strength_slider"), key="strength_slider", step=0.5,
            help="Wie weit die Anomalien im Faktorraum vom Normalen entfernt sind. Bei 3 / 4 / 6 / 9 / 12: AUC des Isolation Forest 0.96 / 0.99 / 1.00 / 1.00 / 1.00 (F1 0.66 / 0.85 / 0.96 / 0.99 / 1.00), "
                 "robust 0.80 / 0.93 / 1.00 (Recall 0.23 / 0.62 / 0.97), klassisch 0.79 / 0.88 / 0.95 (Recall 0.12 / 0.24 / 0.43).",
        )
        st.session_state["_strength_kept"] = strength
    else:
        strength = float(st.session_state.get("_strength_kept", C.DEFAULT_STRENGTH))
        st.session_state["strength_slider"] = strength

    st.markdown("**Isolation Forest**")
    n_trees = st.slider(
        "Bäume", *bounds("trees_slider"), key="trees_slider", step=10,
        help="Anzahl der Isolationsbäume. Die AUC ist schon bei 10 Bäumen 1.00; der F1 mit Schwelle 0.5 ist 0.94 / 0.95 / 0.96 / 0.96 / 0.96 / 0.96 bei 10 / 25 / 50 / 100 / 200 / 500. Die Streuung des F1 über Wald-Seeds "
             "sinkt von 0.037 bei 10 über 0.015 bei 50 auf 0.009 bei 200 Bäumen.",
    )
    psi_hi = psi_max(int(n_tours))
    psi = st.slider(
        "Unterstichprobe ψ", C.PSI_MIN, psi_hi, key="psi_slider", step=1,
        help="Touren je Baum (höchstens die Tourenzahl). Bei ψ = 16 / 32 / 64 / 128 / 256 hat der Isolation Forest F1 0.56 / 0.71 / 0.83 / 0.91 / 0.96 mit Schwelle 0.5 (Fehlalarmrate 0.18 / 0.10 / 0.05 / 0.02 / 0.01), die AUC bleibt 0.99-1.00 - "
             "der Score ist über verschiedene ψ nicht vergleichbar. Gegen Masking hilft ein kleines ψ: bei 30 % dichter Gruppe AUC 0.84 (ψ = 16) gegen 0.70 (ψ = 256).",
    )
    threshold_kind = st.selectbox(
        "Schwelle", C.THRESHOLD_KINDS, key="threshold_kind_select", format_func=lambda k: C.THRESHOLD_LABELS[k],
        help="Standard: Isolation Forest markiert Score über der Schwelle (nominell 0.5), klassisch und robust markieren über dem χ²-Quantil. Erwarteter Anteil: bei allen drei werden die größten Werte markiert - "
             "so entscheidet nur die Rangfolge, aber der Anteil muss bekannt sein.",
    )
    if threshold_kind == "standard":
        cutoff = st.slider(
            "Score-Schwelle (Isolation Forest)", *bounds("cutoff_slider"), key="cutoff_slider", step=0.01,
            help="Ab welchem Anomalie-Wert eine Tour markiert wird. Bei 0.45 / 0.5 / 0.55 / 0.6 / 0.65: F1 0.82 / 0.96 / 0.97 / 0.83 / 0.55, Recall 1.00 / 0.99 / 0.94 / 0.71 / 0.38, Fehlalarmrate 0.050 / 0.009 / 0.001 / 0 / 0 (Standardfall).",
        )
        quantile = st.slider(
            "Schwelle: χ²-Quantil (klassisch, robust)", *bounds("quantile_slider"), key="quantile_slider", step=0.001, format="%.3f",
            help="Ab welchem Anteil der χ²-Verteilung eine Tour bei den Schätzern der Wurzel als Anomalie gilt. Bei 0.9 / 0.95 / 0.975 / 0.99 / 0.999: F1 der robusten Schätzung 0.67 / 0.77 / 0.84 / 0.89 / 0.90, "
                 "Recall der klassischen 0.77 / 0.61 / 0.43 / 0.30 / 0.08.",
        )
        st.session_state["_cutoff_kept"] = cutoff
        st.session_state["_quantile_kept"] = quantile
        share = int(st.session_state.get("_share_kept", C.DEFAULT_SHARE))
        st.session_state["share_slider"] = share                                        # ausgeblendet: der Wert bleibt im Widget-Zustand erhalten
    else:
        share = st.slider(
            "Angenommener Anteil der Anomalien [%]", *bounds("share_slider"), key="share_slider",
            help="Wie viele Touren als Anomalie markiert werden (die größten Werte, für alle drei Detektoren). Beim wahren Anteil 10 % ist der F1 bei angenommenen 2 / 5 / 10 / 20 / 40 % für den Isolation Forest "
                 "0.33 / 0.67 / 0.97 / 0.67 / 0.40, für die robuste Schätzung 0.33 / 0.67 / 0.90 / 0.66 / 0.40, für die klassische 0.32 / 0.56 / 0.69 / 0.58 / 0.39.",
        )
        st.session_state["_share_kept"] = share
        cutoff = float(st.session_state.get("_cutoff_kept", C.DEFAULT_CUTOFF))
        quantile = float(st.session_state.get("_quantile_kept", C.DEFAULT_QUANTILE))
        st.session_state["cutoff_slider"], st.session_state["quantile_slider"] = cutoff, quantile
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren und die Anomalien.")

sync_query_params({
    "n_tours_slider": int(n_tours), "p_slider": int(p_features), "n_noise_slider": int(n_noise), "n_modes_slider": int(n_modes), "curvature_slider": float(curvature), "noise_slider": float(noise),
    "contamination_slider": int(contamination), "kind_select": kind, "strength_slider": float(strength), "trees_slider": int(n_trees), "psi_slider": int(psi), "threshold_kind_select": threshold_kind,
    "cutoff_slider": float(cutoff), "quantile_slider": float(quantile), "share_slider": int(share), "seed_input": int(seed),
})

data_params = (int(n_tours), int(p_features), int(n_noise), int(n_modes), float(curvature), float(round(noise, 2)), int(contamination), kind, float(strength), int(seed))
settings = Settings(n_trees=int(n_trees), psi=int(psi), threshold_kind=threshold_kind, cutoff=float(round(cutoff, 2)), quantile=float(quantile), share=int(share))
with st.spinner("Baue den Wald..."):
    a = _analysis(data_params, settings)
level, code, vd = verdict(a)
ds = a.ds
ifs, cs, rs = a.scores["iforest"], a.scores["classical"], a.scores["robust"]
n_anom = int(ds.anomaly.sum())
n_total = ds.p + ds.n_noise
base_data = tuple(sorted({"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(round(noise, 2)),
                          "contamination": int(contamination), "kind": kind, "strength": float(strength)}.items()))
data_key = data_params + (settings,)

# --- Isolation Forest in Aktion ---------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Isolation Forest in Aktion")
STEP_LABELS = {1: "1 · Touren", 2: "2 · Zufällige Schnitte", 3: "3 · Pfadlängen", 4: "4 · Score und Schwelle", 5: "5 · Ergebnis"}
if "isf_step" not in st.session_state or st.session_state.get("isf_step_owner") != data_key:
    st.session_state["isf_step"] = 1
    st.session_state["isf_step_owner"] = data_key
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="isf_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()

P, axes = projection(ds.X, a.robust)
flag_i = a.flags["iforest"]
mean_paths = a.paths.mean(axis=0)
c_psi = float(isf.c_factor(a.forest.psi))
scores_if = a.values["iforest"]
i_anom = int(np.argmax(np.where(ds.anomaly, scores_if, -np.inf)))
i_norm = int(np.argmin(np.where(~ds.anomaly, np.abs(P).sum(axis=1), np.inf)))         # eine Tour mitten im Normalbereich
tree2d = isf.build_tree(P, np.random.default_rng([seed, 99]), isf.height_limit(len(P)))
segments = isf.tree_segments(tree2d, (P[:, 0].min() - 0.5, P[:, 0].max() + 0.5, P[:, 1].min() - 0.5, P[:, 1].max() + 0.5))
steps_c = sorted({1, 2, 5, 10, 20, 50, 100, 200, 500} & set(range(1, a.paths.shape[0] + 1)) | {a.paths.shape[0]})
conv_ids = list(np.flatnonzero(ds.anomaly)[:6]) + list(np.flatnonzero(~ds.anomaly)[:6])


def _path_text(idx, label):
    steps_p, size = isf.point_path(tree2d, P[idx])
    axis = ("Hauptrichtung 1", "Hauptrichtung 2")
    lines = [f"{k + 1}. {axis[f]} {'<' if left else '≥'} {t:.2f}" for k, (f, t, left) in enumerate(steps_p)]
    return f"**{label}** - allein nach **{len(steps_p)}** Schnitten" + (f" (Blatt mit {size} Touren)" if size > 1 else "") + ":\n\n" + "\n".join(f"- {line}" for line in lines[:12]) + ("\n- ..." if len(lines) > 12 else "")


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns(2)
            c1.markdown("**Die Touren in der Ebene ihrer größten Streuung** (rote Rauten = Sonderfahrten)")
            c1.plotly_chart(build_scatter(P, ds.anomaly), width="stretch", key="step_scatter")
            c2.markdown(f"**Zwei Rohmerkmale: {ds.names[0]} gegen {ds.names[min(1, ds.p - 1)]}** (Einheiten wie gemessen)")
            c2.plotly_chart(build_features(ds.X, ds.anomaly, ds.names, 0, 1), width="stretch", key="step_features")
        elif current_step == 2:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Ein Isolationsbaum auf den zwei Hauptrichtungen** (Illustration: derselbe Algorithmus, aber nur auf diesen zwei Richtungen; Linien = Schnitte, dunkler = früher; Kreise = die zwei Touren rechts)")
            c1.plotly_chart(build_tree_plane(P, ds.anomaly, segments, [(i_norm, "normale Tour", BLUE_TXT), (i_anom, "Sonderfahrt", RED_TXT)]), width="stretch", key="step_tree")
            c2.markdown(_path_text(i_anom, "Sonderfahrt (rot)"))
            c2.markdown(_path_text(i_norm, "Normale Tour (blau)"))
            st.markdown(f"**Pfadlängen derselben zwei Touren in allen {a.paths.shape[0]} Bäumen des echten Waldes (alle {n_total} Merkmale)**")
            st.plotly_chart(build_path_lengths(a.paths[:, i_norm], a.paths[:, i_anom], c_psi), width="stretch", key="step_paths")
        elif current_step == 3:
            c1, c2 = st.columns(2)
            c1.markdown("**Mittlere Pfadlänge je Tour** (über alle Bäume; gestrichelt: c(ψ))")
            c1.plotly_chart(build_mean_paths(mean_paths, ds.anomaly, c_psi), width="stretch", key="step_mean_paths")
            c2.markdown("**Anomalie-Wert nach den ersten k Bäumen** (sechs Sonderfahrten rot, sechs normale Touren blau)")
            conv = isf.score_convergence(a.paths[:, conv_ids], a.forest.psi, steps_c)
            c2.plotly_chart(build_convergence(steps_c, conv, ds.anomaly[conv_ids]), width="stretch", key="step_convergence")
        elif current_step == 4:
            c1, c2 = st.columns(2)
            c1.markdown("**Anomalie-Wert (Score)** der normalen und der Sonderfahrten, mit der Schwelle")
            c1.plotly_chart(build_scores(scores_if, ds.anomaly, ifs["threshold"], "Anomalie-Wert des Isolation Forest"), width="stretch", key="step_scores")
            c2.markdown("**Als Anomalie markierte Touren** (Kreise) in der Ebene der Hauptrichtungen")
            c2.plotly_chart(build_scatter(P, ds.anomaly, flag_i), width="stretch", key="step_flagged")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Kennzahlen bei der gewählten Schwelle**")
            c1.plotly_chart(build_method_bars(a.scores), width="stretch", key="step_bars")
            c2.markdown("**ROC-Kurven** (unabhängig von der Schwelle)")
            curves = {d: (*roc_curve(a.values[d], ds.anomaly), a.scores[d]["auc"]) for d in ("iforest", "classical", "robust")}
            c2.plotly_chart(build_roc(curves), width="stretch", key="step_roc")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"{ds.n} Touren mit {n_total} Kennzahlen" + (f" ({ds.n_noise} davon reines Rauschen)" if ds.n_noise else "") + f", davon {n_anom} Sonderfahrten ({n_anom / ds.n:.0%}; {C.KIND_LABELS[ds.kind]}), "
               f"{ds.n_modes} Betriebsart{'en' if ds.n_modes > 1 else ''}. Die Ebene ist die der zwei größten Streuungsrichtungen der robusten Schätzung der Wurzel in standardisierten Kennzahlen; der Isolation Forest arbeitet mit allen Kennzahlen einzeln.")
elif step == 2:
    st.caption(f"Jeder Schnitt wählt ein Merkmal und einen Schnittpunkt zufällig zwischen kleinstem und größtem Wert im Knoten. Die Sonderfahrt liegt abseits: sie ist im Baum oben nach {len(isf.point_path(tree2d, P[i_anom])[0])} Schnitten allein, "
               f"die normale Tour braucht {len(isf.point_path(tree2d, P[i_norm])[0])}. Im echten Wald mit {n_total} Merkmalen: mittlere Pfadlänge {a.paths[:, i_anom].mean():.1f} gegen {a.paths[:, i_norm].mean():.1f} (c(ψ) = {c_psi:.1f}, ψ = {a.forest.psi}).")
elif step == 3:
    st.caption(f"Mittlere Pfadlänge der Sonderfahrten {mean_paths[ds.anomaly].mean():.1f}, der normalen Touren {mean_paths[~ds.anomaly].mean():.1f} (c(ψ) = {c_psi:.1f}). Mit mehr Bäumen wird der Anomalie-Wert stabil: "
               "die Rangfolge steht schon nach wenigen Bäumen, die Feinheit (und damit die Entscheidung an der Schwelle) braucht mehr.")
elif step == 4:
    st.caption(f"Schwelle {ifs['threshold']:.2f} " + ("(nominell 0.5: alles Normale liegt darum)" if settings.threshold_kind == "standard" else f"(die größten {int(flag_i.sum())} Werte = angenommener Anteil {settings.share} %)")
               + f". Der mittlere Anomalie-Wert der normalen Touren ist {scores_if[~ds.anomaly].mean():.2f}, der der Sonderfahrten {scores_if[ds.anomaly].mean():.2f}; markiert wurden {int(flag_i.sum())} Touren "
               f"(Recall {_pct(ifs['recall'])}, Fehlalarmrate {ifs['false_alarm']:.1%}).")
else:
    st.caption("Die ROC-Kurve zeigt die Rangfolge (AUC), die Balken die Wirkung der Schwelle: eine perfekte Rangfolge kann trotzdem viele Fehlalarme oder verpasste Anomalien haben, wenn die Schwelle nicht passt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was der Isolation Forest gefunden hat – gegen die Wurzel")
st.caption(
    "Anomalie = Tour über der Schwelle (Standard: Score über der Score-Schwelle beim Isolation Forest, χ²-Quantil bei klassisch und robust). **AUC**: Wahrscheinlichkeit, dass eine zufällige Sonderfahrt einen größeren Wert hat als eine "
    "zufällige normale Tour (1 = perfekte Rangfolge, 0.5 = Raten, darunter: die Anomalien wirken normaler als die Normalen). **Recall**: Anteil der gefundenen Sonderfahrten. **Fehlalarmrate**: Anteil der normalen Touren, die markiert werden."
)
best_root = vd["best_root_auc"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC (Isolation Forest)", f"{ifs['auc']:.2f}", delta=f"beste Wurzel {best_root:.2f}", delta_color="off", help="Rangfolge der Anomalie-Werte; im Delta die bessere AUC von klassisch und robust.")
m2.metric("Recall (Isolation Forest)", _pct(ifs["recall"]), delta=f"klassisch {_pct(cs['recall'])} · robust {_pct(rs['recall'])}", delta_color="off", help=f"Anteil der {n_anom} Sonderfahrten, die bei der Schwelle markiert werden.")
m3.metric("Fehlalarmrate (Isolation Forest)", f"{ifs['false_alarm']:.1%}", delta=f"klassisch {cs['false_alarm']:.1%} · robust {rs['false_alarm']:.1%}", delta_color="off",
          help="Anteil der normalen Touren, die als Anomalie markiert werden.")
m4.metric("F1 (Isolation Forest)", f"{ifs['f1']:.2f}", delta=f"mit bekanntem Anteil {a.oracle_f1:.2f}", delta_color="off",
          help="Harmonisches Mittel aus Precision und Recall bei der Schwelle; darunter der F1, den dieselbe Rangfolge erreichte, wenn der wahre Anteil bekannt wäre (die größten Werte markiert).")

_t = vd
_caveat = ""
if _t["iforest_f1"] < a.oracle_f1 - 0.15 and code == "if_wins":
    _caveat = (f" Bedenken: die Schwelle passt nicht - mit ihr hat der Isolation Forest F1 {_t['iforest_f1']:.2f} (Recall {_pct(_t['iforest_recall'])}, Fehlalarmrate {_t['iforest_false_alarm']:.1%}), "
               f"mit bekanntem Anteil wären es {a.oracle_f1:.2f}: die Rangfolge trägt, die feste Schwelle nicht.")
if code == "if_wins":
    st.success(f"✅ Der Isolation Forest trennt besser: AUC {_t['iforest_auc']:.2f} gegen {_t['classical_auc']:.2f} (klassisch) und {_t['robust_auc']:.2f} (robust). Er braucht weder eine Verteilungsannahme noch mehr Touren als Merkmale.{_caveat}")
elif code == "comparable":
    st.success(f"✅ Die Rangfolge ist gleich gut (AUC Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}, klassisch {_t['classical_auc']:.2f}); bei der Schwelle: F1 {_t['iforest_f1']:.2f} (Isolation Forest), "
               f"{_t['robust_f1']:.2f} (robust), {_t['classical_f1']:.2f} (klassisch).")
elif code == "dense_masking":
    st.warning(f"⚠️ Dichte Gruppe: die robuste Schätzung der Wurzel ist besser (AUC {_t['robust_auc']:.2f} gegen {_t['iforest_auc']:.2f}). Die Gruppe ({_t['contamination']:.0f} % der Touren) ist auch in der Unterstichprobe dicht und lässt sich "
               f"schwerer isolieren; mit F1 {_t['iforest_f1']:.2f} bei {_t['iforest_false_alarm']:.0%} Fehlalarmen. Ein kleineres ψ hilft (siehe Experiment Masking); die robuste Schätzung stützt sich dagegen auf die dichtere Hälfte.")
elif code == "root_wins":
    st.warning(f"⚠️ Die robuste Schätzung der Wurzel trennt besser: AUC {_t['robust_auc']:.2f} gegen {_t['iforest_auc']:.2f}.")
elif code == "gap":
    st.warning(f"⚠️ Die Anomalien liegen in der Lücke zwischen den Betriebsarten: AUC {_t['iforest_auc']:.2f} beim Isolation Forest, {_t['robust_auc']:.2f} robust, {_t['classical_auc']:.2f} klassisch. Als dichte Gruppe zwischen den Betriebsarten "
               f"sind sie selbst ein weiterer Modus - F1 nur {_t['iforest_f1']:.2f}. Bei wenigen, verstreuten Anomalien in der Lücke (2 %) findet der Isolation Forest sie besser (AUC 0.81).")
elif code == "threshold_off":
    st.warning(f"⚠️ Die Schwelle passt nicht: die Rangfolge ist perfekt (AUC {_t['iforest_auc']:.2f}), aber bei der gewählten Schwelle ist F1 {_t['iforest_f1']:.2f} (Recall {_pct(_t['iforest_recall'])}, Fehlalarmrate {_t['iforest_false_alarm']:.1%}); "
               f"mit dem wahren Anteil wären es {a.oracle_f1:.2f}. Der Isolation Forest hat keine kalibrierte Fehlalarmrate - anders als das χ²-Quantil der Wurzel unter der Gauß-Annahme.")

d1, d2c = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    rows = [("AUC", "auc", "{:.2f}"), ("mittlere Präzision (AP)", "ap", "{:.2f}"), ("Precision", "precision", "{:.2f}"), ("Recall", "recall", "{:.2f}"), ("F1", "f1", "{:.2f}"),
            ("Fehlalarmrate", "false_alarm", "{:.3f}"), ("Schwelle", "threshold", "{:.2f}")]
    st.table({"Kennzahl": [r[0] for r in rows], "Isolation Forest": [r[2].format(ifs[r[1]]) for r in rows], "klassisch": [r[2].format(cs[r[1]]) for r in rows], "robust (MCD)": [r[2].format(rs[r[1]]) for r in rows]})
with d2c:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Rechenzeit", "Bäume × ψ", "Merkmale", "mittlerer Score (normal / Anomalie)"],
              "Isolation Forest": [f"{a.seconds['iforest'] * 1000:.0f} ms", f"{len(a.forest.trees)} × {a.forest.psi}", f"{n_total} (einzeln)", f"{scores_if[~ds.anomaly].mean():.2f} / {scores_if[ds.anomaly].mean():.2f}"],
              "robust (MCD)": [f"{a.seconds['robust'] * 1000:.0f} ms", f"h = {a.robust.h}", f"{n_total} (gemeinsam)", f"d² {a.values['robust'][~ds.anomaly].mean():.1f} / {a.values['robust'][ds.anomaly].mean():.1f}"]})
    st.caption("Der Isolation Forest bewertet jedes Merkmal einzeln durch Schnitte, die Wurzel alle gemeinsam über die Kovarianz. Der Score ist skaleninvariant, aber nicht drehungsinvariant (achsenparallele Schnitte).")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Touren, Merkmalen, Anomalien und Wald ab?")
sweep_options = [k for k in SWEEP_LABELS if not ((kind == "gap" and k in ("strength", "n_modes")) or (threshold_kind == "share" and k in ("cutoff", "quantile")) or (threshold_kind == "standard" and k == "share"))]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
current = {"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(noise), "contamination": int(contamination),
           "strength": float(strength), "n_trees": int(n_trees), "psi": int(psi), "cutoff": float(cutoff), "quantile": float(quantile), "share": int(share)}[sweep_param]
if st.button("Sweep über 5 feste Datensätze berechnen (dauert einige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, data_key)}
if (sweep_param, data_key) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
        rows_sweep = _sweep(sweep_param, tuple(kv for kv in base_data if kv[0] != sweep_param), settings)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], current=current), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste. Links die Rangfolge (AUC), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle: "
               "die Rangfolge und die Entscheidung an der Schwelle sind zwei Fragen. Regler des Isolation Forest (Bäume, Unterstichprobe, Schwelle) verändern die Linien der Wurzel nicht.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wald-Parameter: Bäume und Unterstichprobe")
if st.button("Bäume und ψ durchfahren (dauert etwa 25 Sekunden)", key="parameter_start"):
    st.session_state["parameter_on"] = True
if st.session_state.get("parameter_on"):
    with st.spinner("Rechne Bäume × ψ × 5 Datensätze und die Streuung über Wald-Seeds..."):
        pt = _parameters(base_data, settings)
    st.plotly_chart(build_parameters(pt["trees"], pt["psi"]), width="stretch", key="parameter_chart")
    st.table({"Bäume": [r["n_trees"] for r in pt["spread"]], "Streuung der AUC über 6 Wald-Seeds": [f"{r['auc_std']:.4f}" for r in pt["spread"]], "Streuung des F1 bei der Schwelle": [f"{r['f1_std']:.3f}" for r in pt["spread"]]})
    st.caption("Mittel über 5 feste Datensätze. Die Rangfolge (AUC) ist schon bei wenigen Bäumen stabil, die Entscheidung an der Schwelle streut mit wenigen Bäumen deutlich (Tabelle). Kleine Unterstichproben verschieben den Score: "
               "bei fester Schwelle 0.5 fällt der F1 mit ψ, obwohl die AUC gleich bleibt - der Score ist zwischen verschiedenen ψ nicht vergleichbar.")

st.markdown("---")

st.subheader("🔬 Geister-Regionen: was achsenparallele Schnitte anrichten")
if st.button("Score-Karte und Anisotropie berechnen (dauert etwa 6 Sekunden)", key="ghost_start"):
    st.session_state["ghost_on"] = True
if st.session_state.get("ghost_on"):
    with st.spinner("Berechne die Score-Karte..."):
        X2, gx, gy, gS, gell, g_aniso, g_rows = _ghost(tuple(kv for kv in base_data if kv[0] not in ("n_noise", "p", "contamination", "kind", "strength")), settings, int(seed))
    c1, c2 = st.columns([3, 2])
    c1.markdown("**Score-Karte** eines Isolation Forest über die ersten zwei Merkmale (ohne Anomalien; dunkler = normaler; gestrichelt: robuste Ellipse)")
    c1.plotly_chart(build_ghost_map(gx, gy, gS, X2, ds.names[:2], gell), width="stretch", key="ghost_map")
    c2.markdown("**Anisotropie**: Score der Ecken minus der Korridore")
    c2.plotly_chart(build_anisotropy(g_rows), width="stretch", key="ghost_aniso")
    st.caption(f"Für diese Aufnahme: Anisotropie {g_aniso:.3f} bei 1 σ Abstand. Rasterpunkte im gleichen Abstand zum nächsten Datenpunkt bekommen verschiedene Scores, je nachdem ob sie **neben dem Datenbereich** eines Merkmals liegen (Korridor: die Schnitte auf dem anderen Merkmal "
               "reichen dort nicht zur Isolation) oder außerhalb beider Bereiche (Ecke): die Ecken sind um 0.06 bis 0.10 auffälliger. Die Ellipse der Wurzel ist dagegen isotrop im Abstand. Diese Bänder sind die 'Geister' - die Schwäche, "
               "an der der Extended Isolation Forest (schräge Schnitte) ansetzt. Nur die zwei ersten Merkmale, Betriebsarten wie in der Seitenleiste.")

st.markdown("---")

st.subheader("🔬 Die Schwächen der Wurzel: Lücke, Betriebsarten, Krümmung")
if st.button("Betriebsarten × Art der Anomalien vergleichen (dauert etwa 10 Sekunden)", key="modes_start"):
    st.session_state["modes_on"] = True
if st.session_state.get("modes_on"):
    with st.spinner("Vergleiche 1-3 Betriebsarten × 3 Arten × 5 Datensätze..."):
        mt = _modes(tuple(kv for kv in base_data if kv[0] not in ("kind", "n_modes")), settings)
    st.plotly_chart(build_modes(mt), width="stretch", key="modes_chart")
    st.caption("AUC der drei Detektoren (Mittel über 5 feste Datensätze). Verstreute Anomalien findet der Isolation Forest bei jeder Zahl von Betriebsarten (AUC 1.00), wo die robuste Schätzung nachlässt; "
               "bei einer dichten Gruppe abseits ist die robuste Schätzung bei einer und zwei Betriebsarten besser (1.00 gegen 0.95 und 0.98), bei drei Betriebsarten der Isolation Forest (0.95 gegen 0.78); Anomalien in der Lücke findet keiner zuverlässig (drei Betriebsarten: Isolation Forest 0.27, unter Raten). Der Krümmungs-Sweep oben zeigt: gekrümmter Normalbereich ändert die Rangfolge nicht, nur die χ²-Schwelle.")

st.markdown("---")

st.subheader("🔬 Viele Merkmale, Rauschmerkmale, wenige Touren")
if st.button("Rauschmerkmale und Tourenzahl × Merkmalszahl durchfahren (dauert etwa 45 Sekunden)", key="dimension_start"):
    st.session_state["dimension_on"] = True
if st.session_state.get("dimension_on"):
    with st.spinner("Rechne 6 Tourenzahlen × 5 Merkmalszahlen × 5 Datensätze und die Rauschmerkmale..."):
        dt, nt = _dimension(tuple(kv for kv in base_data if kv[0] not in ("n", "p", "n_noise")), settings)
    st.markdown("**Rauschmerkmale 0 bis 40**")
    st.plotly_chart(build_sweep(nt, SWEEP_LABELS["n_noise"]), width="stretch", key="noise_chart")
    st.markdown("**Tourenzahl × Merkmalszahl**")
    st.plotly_chart(build_dimension(dt), width="stretch", key="dimension_chart")
    st.caption("Mittel über 5 feste Datensätze. Die AUC des Isolation Forest bleibt bei jeder Tourenzahl und Merkmalszahl bei 0.99-1.00, auch wenn nicht mehr Touren als Merkmale da sind (die robuste Schätzung liegt dort im Zufallsbereich, um 0.35-0.6, je nach Rechner). "
               "Rauschmerkmale kosten kaum Rangfolge, aber die Entscheidung an der festen Schwelle 0.5 (F1 rechts oben): der Anomalie-Wert der Sonderfahrten rückt an 0.5 heran.")

st.markdown("---")

st.subheader("🔬 Die Schwelle: Score-Schwelle gegen erwarteten Anteil")
if st.button("Schwellen vergleichen (dauert etwa 15 Sekunden)", key="threshold_start"):
    st.session_state["threshold_on"] = True
if st.session_state.get("threshold_on"):
    with st.spinner("Rechne 5 Schwellen × 5 Datensätze und drei angenommene Anteile..."):
        tt = _threshold(base_data, settings)
    c1, c2 = st.columns(2)
    c1.markdown("**Kennzahlen des Isolation Forest je Score-Schwelle**")
    c1.plotly_chart(build_cutoff(tt["cutoff"]), width="stretch", key="cutoff_chart")
    c2.markdown("**F1 bei falsch angenommenem Anteil** (½×, 1×, 2× des wahren)")
    c2.plotly_chart(build_wrong_share(tt["wrong_share"]), width="stretch", key="wrong_share_chart")
    st.table({"Tourenzahl": [r["n"] for r in tt["normal_scores"]], "mittlerer Score der normalen Touren": [f"{r['median']:.3f}" for r in tt["normal_scores"]]})
    st.caption("Links: die Score-Schwelle 0.5 ist eine Faustregel - F1 ist bei 0.5-0.55 am besten und fällt zu beiden Seiten. Rechts: ein falscher angenommener Anteil kostet alle drei Detektoren gleich viel, denn dann entscheidet nur die Rangfolge, "
               "die aber bei allen (fast) perfekt ist. Tabelle: der mittlere Score der Normalen hängt von der Tourenzahl ab (bei kleinen Stichproben höher) - die nominelle Schwelle 0.5 ist nur ungefähr.")

st.markdown("---")

st.subheader("🔬 Masking: wenn die Anomalien eine dichte Gruppe bilden")
if st.button("Anteil der dichten Gruppe von 2 bis 45 % durchfahren (dauert etwa 20 Sekunden)", key="masking_start"):
    st.session_state["masking_on"] = True
if st.session_state.get("masking_on"):
    with st.spinner("Rechne 10 Anteile × 5 Datensätze und 5 Unterstichproben..."):
        mk = _masking(tuple(kv for kv in base_data if kv[0] not in ("kind", "contamination", "n_modes")), settings)
    st.plotly_chart(build_masking(mk), width="stretch", key="masking_chart")
    st.caption("Mittel über 5 feste Datensätze, eine Betriebsart. Der Isolation Forest verliert früher als die robuste Schätzung (AUC 0.85 gegen 1.00 bei 20 %), ist aber besser als die klassische (0.62), und bei 30 % ist er der einzige der drei deutlich über Raten (0.70 gegen 0.52 klassisch und 0.49 robust). "
               "Rechts: eine kleinere Unterstichprobe hilft (bei 30 %: 0.84 mit ψ = 16, 0.70 mit ψ = 256) - Liu et al. begründen damit die kleinen ψ. Ab etwa 40 % versagen alle: die Gruppe ist dann der Normalbereich.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Anomalien sind wenige und verstreut** | Eine dichte Gruppe von 20 % senkt die AUC auf 0.85 (robuste Schätzung der Wurzel 1.00), bei 40 % auf 0.40. Anomalien in der Lücke zwischen zwei Betriebsarten: AUC 0.54, F1 0.02. | kleineres ψ (hilft, löst es nicht); lokale Dichte (**LOF**) |
| **Achsenparallele Schnitte genügen** | Rasterpunkte im gleichen Abstand zum Datenrand bekommen je nach Lage verschiedene Scores: Punkte in den Ecken sind um 0.06 bis 0.10 auffälliger als solche neben dem Datenbereich - Bänder ("Geister") entlang der Achsen. | **Extended Isolation Forest** (schräge Schnitte) |
| **Der Score ist eine Schwelle** | Die Rangfolge ist fast perfekt, die feste Schwelle 0.5 nicht: bei 40 Rauschmerkmalen Recall 0.39 (F1 0.56, mit bekanntem Anteil 0.84), bei 20 Touren Fehlalarmrate 0.156, bei ψ = 16 Fehlalarmrate 0.18. Ein falsch angenommener Anteil (½× oder 2×) senkt F1 von 0.97 auf 0.67. | Kalibrierung; parameterfreie Verfahren (**ECOD**) |
| **Irrelevante Merkmale stören nicht** | Die AUC bleibt bei 40 Rauschmerkmalen bei 0.99 (robuste Schätzung 0.88), aber jeder Schnitt auf ein Rauschmerkmal ist verschenkt: die Scores der Anomalien rücken an 0.5. | Ensembles über Merkmalsteilmengen (**Feature Bagging**) |
| **Der Wald ist groß genug** | Die AUC ist schon bei 10 Bäumen 1.00; der F1 an der Schwelle streut über Wald-Seeds mit 0.037 (10 Bäume), 0.015 (50) und 0.009 (200). | mehr Bäume (Rechenzeit) |
| **Ein Gauß'scher Normalbereich (Wurzel) gilt hier nicht** | Der Isolation Forest kommt ohne aus: bei gekrümmtem Normalbereich F1 0.95-0.98 (robuste Schätzung mit χ²-Schwelle 0.38-0.49); bei n < p AUC 1.00, wo die Wurzel nichts entscheiden kann. | (das ist der Gewinn dieses Astes) |
"""
)
st.caption(
    "Die Nachbarn der Anomalie-Erkennung-Linie: die Wurzel Elliptic Envelope (gebaut), Extended Isolation Forest (die Fortsetzung dieses Astes, noch nicht gebaut), LOF und Feature Bagging, One-Class SVM und Deep SVDD, ECOD und ein Autoencoder. "
    "Keiner ist überlegen: der Isolation Forest ist schnell, robust gegen die Annahmen der Wurzel und bleibt dennoch ein Schwellenproblem."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Isolationsbaum.** Auf einer Unterstichprobe $S$ mit $|S| = \psi$: Knoten mit Punktmenge $Q$: wähle $q$ zufällig unter den nicht konstanten Merkmalen und $t \sim U(\min_{x \in Q} x_q, \max_{x \in Q} x_q)$; $Q_L = \{x: x_q < t\}$, $Q_R = Q \setminus Q_L$.
Abbruch bei $|Q| = 1$, gleichen Punkten oder Tiefe $\lceil \log_2 \psi \rceil$.

**Pfadlänge.** $h(x) = e + c(|Q_\text{Blatt}|)$ mit $e$ = Zahl der Schnitte bis zum Blatt; $c(n) = 2\,(\ln(n-1) + \gamma) - 2(n-1)/n$ für $n > 2$, $c(2) = 1$, $c(1) = 0$ ($\gamma$ Euler-Konstante): die mittlere Pfadlänge einer erfolglosen Suche in einem binären Suchbaum über $n$ Punkte.

**Anomalie-Wert.** $s(x, \psi) = 2^{-E[h(x)] / c(\psi)}$, $E$ über die Bäume des Waldes. $E[h] \to 0$: $s \to 1$; $E[h] = c(\psi)$: $s = 0{,}5$; $E[h] \to \psi - 1$: $s \to 0$.

**Schwelle.** Standard: $s > 0{,}5$ (Faustregel, nicht kalibriert). Alternativ die $\lceil \alpha n \rceil$ größten Werte bei angenommenem Anteil $\alpha$. **Vergleich:** klassisch und robust mit dem Mahalanobis-Abstand $d^2$ und $\chi^2_{p,\,q}$ (siehe elliptic-envelope-demo; MCD als FastMCD).

**Kennzahlen.** AUC (Rangsumme, Bindungen halb), mittlere Präzision, Precision, Recall, F1, Fehlalarmrate. **Score-Anisotropie:** mittlerer Score der Rasterpunkte außerhalb der Wertebereiche beider Merkmale ("Ecken") minus der in mindestens einem Wertebereich ("Korridore"), bei gleichem standardisiertem Abstand zum nächsten Datenpunkt (±0,15).

**Invarianz.** Skalierung und Verschiebung eines Merkmals ändern den Score nicht (der gleichverteilte Schnittpunkt zwischen Minimum und Maximum skaliert mit; per Test bei gleichem Wald-Seed identisch) - Drehungen schon: die Schnitte sind achsenparallel.

**Grenzen.** (1) Dichte Anomaliegruppen sind schwer zu isolieren (Masking). (2) Achsenparallele Schnitte erzeugen Bänder entlang der Achsen. (3) Der Score hat keine Verteilung unter dem Normalmodell, also keine kalibrierte Schwelle. (4) Rauschmerkmale verdünnen die Schnitte.

Implementiert in `isf_algorithm.py` (Isolationsbaum, Wald, Pfadlängen, Score), `isf_ee_algorithm.py` (klassisch und MCD, wortgleich aus der Wurzel), `isf_scenario.py` (Touren mit Betriebsarten, Krümmung, Anomalien, Rauschmerkmalen),
`isf_evaluation.py` (Kennzahlen, Sweeps, Experimente, Anisotropie, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
