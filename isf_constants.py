"""Defaults, Slider-Grenzen und Presets für die Isolation-Forest-Demo (Anomalie-Erkennung an Lieferrouten-Kennzahlen; Szenario und Vergleichsschätzer aus der Elliptic-Envelope-Demo)."""

# --- Merkmale: die 12 Kennzahlen der PCA-Demo (Name, Einheit, Mittelwert, typische Streuung), dazu Zusatzmerkmale für den Fall n < p --------------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_BASE_FEATURES = len(FEATURES)
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_BASE_FEATURES))
# Reihenfolge, in der die ersten p Merkmale gewählt werden: reihum durch die vier Gruppen, damit schon p = 2 beide latenten Faktoren sieht (Distanz und Zeitfenster-Enge)
FEATURE_ORDER = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)
EXTRA_MEAN, EXTRA_SCALE = 50.0, 10.0                   # Zusatzmerkmale 13 ... p (zufällige Mischungen der latenten Faktoren plus eigenes Rauschen)

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 20, 600
DEFAULT_P = 12
DEFAULT_N_NOISE = 0
N_NOISE_MIN, N_NOISE_MAX = 0, 40
P_MIN, P_MAX = 2, 30
N_MODES_MIN, N_MODES_MAX = 1, 3
DEFAULT_N_MODES = 1
DEFAULT_CURVATURE = 0.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_CONTAMINATION = 10                             # Prozent
CONTAMINATION_MIN, CONTAMINATION_MAX = 1, 45
KINDS = ("scattered", "cluster", "gap")
KIND_LABELS = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke zwischen den Betriebsarten"}
DEFAULT_KIND = "scattered"
DEFAULT_STRENGTH = 6.0
STRENGTH_MIN, STRENGTH_MAX = 3.0, 12.0
DEFAULT_SUPPORT = 0.5                                   # Stützanteil h/n (0.5 = größter Bruchpunkt); 1.0 = alle Punkte = klassische Schätzung
SUPPORT_MIN, SUPPORT_MAX = 0.5, 1.0
DEFAULT_QUANTILE = 0.975                                # chi^2-Quantil der Schwelle
QUANTILE_MIN, QUANTILE_MAX = 0.90, 0.999
DEFAULT_REWEIGHT = True
DEFAULT_SEED = 7

# --- Isolation Forest ----------------------------------------------------------------------------------------------------
DEFAULT_TREES = 100
TREES_MIN, TREES_MAX = 10, 500
DEFAULT_PSI = 256                                       # Unterstichprobe je Baum (höchstens die Tourenzahl)
PSI_MIN, PSI_MAX = 16, 512
THRESHOLD_KINDS = ("standard", "share")
THRESHOLD_LABELS = {"standard": "Standard (Score 0.5 bzw. χ²-Quantil)", "share": "erwarteter Anteil (für alle drei)"}
DEFAULT_THRESHOLD_KIND = "standard"
DEFAULT_CUTOFF = 0.5                                    # nominelle Score-Schwelle: um 0.5 liegt alles Normale
CUTOFF_MIN, CUTOFF_MAX = 0.40, 0.80
DEFAULT_SHARE = 10                                      # angenommener Anteil der Anomalien [%] (= der wahre im Standardfall)
SHARE_MIN, SHARE_MAX = 1, 45

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
Q = 2                                                   # latente Faktoren (fest; die PCA-Demo variiert sie, hier geht es um Anomalien)
CROSS_LOADING = 0.15
WITHIN_LOADINGS = (0.95, 0.9, 0.85)
CURVATURE_FREQUENCY = 1.6
CURVATURE_AMPLITUDE = 2.0
LAYOUT_SEED = 20240915                                  # dieselben festen Matrizen wie in der PCA-Demo
EXTRA_LAYOUT_SEED = LAYOUT_SEED + 2
MODE_RADIUS = 2.2                                       # Betriebsarten liegen auf einem Kreis dieses Radius im Faktorraum
MODE_SD = 0.6                                           # Streuung innerhalb einer Betriebsart (bei nur einer Betriebsart 1, wie in der PCA-Demo)
CLUSTER_SD = 0.3                                        # Streuung der dichten Anomalie-Gruppe
GAP_SD = 0.3                                            # Streuung der Anomalien in der Lücke
CLUSTER_ANGLE = 0.6                                     # Richtung der dichten Gruppe im Faktorraum (Bogenmaß)

# --- Auswertung --------------------------------------------------------------------------------------------------------
MCD_STARTS = 500                                        # zufällige Startmengen (je zwei C-Schritte)
MCD_KEEP = 10                                           # die besten davon laufen bis zur Konvergenz
MCD_INITIAL_STEPS = 2
MCD_MAX_STEPS = 50
RIDGE = 1e-9                                            # relative Regularisierung der Kovarianz (n < p)
SWEEP_SEEDS = tuple(100_000 + i for i in range(5))

# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(n=DEFAULT_N_TOURS, p=DEFAULT_P, n_noise=DEFAULT_N_NOISE, n_modes=DEFAULT_N_MODES, curvature=DEFAULT_CURVATURE, noise=DEFAULT_NOISE, contamination=DEFAULT_CONTAMINATION,
                kind=DEFAULT_KIND, strength=DEFAULT_STRENGTH, trees=DEFAULT_TREES, psi=DEFAULT_PSI, threshold_kind=DEFAULT_THRESHOLD_KIND, cutoff=DEFAULT_CUTOFF, quantile=DEFAULT_QUANTILE,
                share=DEFAULT_SHARE, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _preset(),
    "Viele Rauschmerkmale (40)": _preset(n_noise=40),
    "Wenige Touren, viele Merkmale": _preset(n=20, p=30, psi=20),
    "Dichte Gruppe abseits (20 %)": _preset(kind="cluster", contamination=20),
    "Anomalien in der Lücke": _preset(n_modes=2, kind="gap"),
    "Falscher erwarteter Anteil": _preset(threshold_kind="share", share=5),
}
PRESET_HELP = {
    "Standardfall": "Im Mittel über fünf Aufnahmen: 300 Touren, 12 Merkmale, 10 % verstreute Anomalien. Die Rangfolge ist bei Isolation Forest und robuster Schätzung perfekt (AUC 1.00, klassisch 0.95); mit der Standardschwelle "
                    "(Score 0.5) erreicht der Isolation Forest F1 0.96 bei 0.9 % Fehlalarmen, die robuste Schätzung F1 0.84 (Fehlalarmrate 3.9 %), die klassische F1 0.57 (Recall 0.43).",
    "Viele Rauschmerkmale (40)": "Im Mittel über fünf Aufnahmen: 40 unabhängige Rauschmerkmale zu den 12 Kennzahlen. Die Rangfolge des Isolation Forest bleibt gut (AUC 0.99; robust 0.88, klassisch 0.79), aber die Anomalie-Werte rücken "
                                 "an 0.5 heran: mit der Schwelle 0.5 findet er nur 39 % der Anomalien (F1 0.56); mit bekanntem Anteil wären es F1 0.84 - die Rangfolge trägt, die feste Schwelle nicht.",
    "Wenige Touren, viele Merkmale": "Im Mittel über fünf Aufnahmen: 20 Touren, 30 Merkmale (n < p): klassisch und robust können nichts entscheiden (AUC im Zufallsbereich um 0.5-0.6, nichts markiert), der Isolation Forest hat AUC 1.00 - "
                                      "aber mit der Schwelle 0.5 markiert er 14 % der Normalen (F1 0.61), weil sein Score bei kleinen Stichproben höher liegt.",
    "Dichte Gruppe abseits (20 %)": "Im Mittel über fünf Aufnahmen: 20 % der Touren bilden eine dichte Gruppe abseits. Der Isolation Forest isoliert sie schlechter, weil sie selbst in der Unterstichprobe dicht ist "
                                    "(AUC 0.85, F1 0.44); die robuste Schätzung stützt sich auf die dichtere Hälfte und findet sie (AUC 1.00, F1 0.94). Klassisch 0.62.",
    "Anomalien in der Lücke": "Im Mittel über fünf Aufnahmen: zwei Betriebsarten, 10 % Anomalien in der Lücke dazwischen. Der Isolation Forest liegt mit AUC 0.54 knapp über Raten (klassisch und robust 0.40, darunter), "
                              "aber praktisch nichts wird gefunden (F1 0.02): die Anomalien sind eine dichte Gruppe, die als dritter Modus gilt.",
    "Falscher erwarteter Anteil": "Im Mittel über fünf Aufnahmen: der angenommene Anteil ist 5 %, der wahre 10 %. Die Rangfolge bleibt perfekt (AUC 1.00), aber es werden nur die 5 % auffälligsten markiert: "
                                  "Recall 0.50, F1 0.67 (mit dem wahren Anteil 0.97). Klassisch F1 0.56, robust 0.67 - wer den Anteil nicht kennt, hat bei allen dreien dasselbe Problem.",
}
# Bänder (Seed des Presets; mit dem ausgelieferten Code kalibriert, bewusst weit): Kennzahlen der Detektoren (iforest_*, classical_*, robust_*) und erlaubte Urteile (verdict)
PRESET_EXPECTED_BANDS = {
    "Standardfall": {"iforest_auc": (0.97, 1.0), "iforest_f1": (0.85, 1.0), "robust_auc": (0.97, 1.0), "classical_recall": (0.2, 0.7), "verdict": ("comparable",)},
    "Viele Rauschmerkmale (40)": {"iforest_auc": (0.95, 1.0), "robust_auc": (0.7, 0.95), "iforest_recall": (0.0, 0.6), "iforest_f1": (0.0, 0.7), "verdict": ("if_wins",)},
    "Wenige Touren, viele Merkmale": {"iforest_auc": (0.95, 1.0), "robust_recall": (0.0, 0.2), "classical_recall": (0.0, 0.2), "iforest_false_alarm": (0.05, 0.3), "verdict": ("if_wins",)},
    "Dichte Gruppe abseits (20 %)": {"iforest_auc": (0.7, 0.97), "robust_auc": (0.95, 1.0), "verdict": ("dense_masking",)},
    "Anomalien in der Lücke": {"iforest_auc": (0.3, 0.8), "robust_auc": (0.0, 0.6), "iforest_recall": (0.0, 0.3), "verdict": ("gap",)},
    "Falscher erwarteter Anteil": {"iforest_auc": (0.97, 1.0), "iforest_f1": (0.5, 0.8), "iforest_recall": (0.3, 0.7), "verdict": ("threshold_off",)},
}
