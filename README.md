# Isolation Forest – Anomalien sind leicht zu isolieren – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-isolation-forest-demo.streamlit.app/)**

Zweites Stück der **Anomalie-Erkennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – den **Isolation Forest** – an einem wachsenden Beispiel, mit den beiden Schätzern der Wurzel
([elliptic-envelope-demo](../elliptic-envelope-demo): klassisch und robust per MCD) als Vergleich. Vehikel: dieselben **Lieferrouten-Kennzahlen** wie in der Wurzel und der [pca-demo](../pca-demo) (Szenario und Schätzer wortgleich übernommen, per Test gegen eingefrorene Werte geprüft);
neu sind **Rauschmerkmale** (0–40 unabhängige Spalten).

**Einordnung in die Reihe (die Kanten des Graphen):** die Wurzel beschreibt das Normale (eine Gauß'sche Wolke) und markiert, was außerhalb liegt. Der Isolation Forest beschreibt **nichts** Normales, sondern fragt, **wie schnell sich eine Tour vom Rest abtrennen lässt**:
zufällige achsenparallele Schnitte auf zufälligen Unterstichproben, Anomalien haben kurze Pfade. Er ist ein **unabhängiger Ast direkt nach der Wurzel** und behebt deren Annahmen (ein Normalbereich, Gauß, viele Touren je Merkmal) auf einem völlig anderen Weg – mit eigenen Schwächen,
die die Fortsetzung **Extended Isolation Forest** (schräge Schnitte) aufgreift. Die Linie hat **keinen Konvergenzpunkt**; die Zufallsbäume sind verwandt mit dem Bagging der Baum-Verfahren (dort noch nicht gebaut).
```
elliptic-envelope-demo (Wurzel: robuste Ellipse)
  ├─ ECOD                       (Kontrast: verteilungsfrei)                       [nicht gebaut]
  ├─ LOF → Feature Bagging      (lokale Dichte; Ensembles gegen viele Merkmale)   [nicht gebaut]
  ├─ One-Class SVM → Deep SVDD  (gelernte Grenze)                                 [nicht gebaut]
  ├─ isolation-forest-demo → Extended Isolation Forest (Zufallsbäume)             [dieses Stück; EIF nicht gebaut]
  └─ Autoencoder                (Rekonstruktionsfehler)                           [nicht gebaut]
```

| Frage | Ergebnis (300 Touren, 12 Merkmale, 10 % verstreute Anomalien im Abstand 6 Faktor-σ, ein Normalbereich, Rauschen 0.25; 100 Bäume, ψ = 256, Score-Schwelle 0.5 bzw. χ²-Quantil 0.975; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Standardfall | ✅ Rangfolge bei Isolation Forest und robuster Schätzung perfekt (AUC 1.00, klassisch 0.95); F1 an der Standardschwelle **0.96** (Fehlalarmrate 0.9 %) gegen 0.84 (robust, 3.9 %) und 0.57 (klassisch, Recall 0.43) |
| Wurzel-Schwächen | ✅ **Gekrümmter Normalbereich**: F1 des Isolation Forest 0.95–0.98, robust mit χ²-Schwelle 0.49–0.38, klassisch 0.90 – *aber* die Rangfolge (AUC 1.00) ist für alle drei gleich, der Unterschied liegt nur an der χ²-Schwelle. **Mehrere Betriebsarten**: Recall 0.99 / 0.99 / 0.97 (robust 0.97 / 0.77 / 0.47). **n < p** (20 Touren, 30 Merkmale): AUC **1.00**, klassisch 0.60, robust 0.55 (beide markieren nichts) |
| Rauschmerkmale | ⚠️ AUC bleibt bei 40 Rauschmerkmalen **0.99** (robust 0.88, klassisch 0.79) – aber der Recall bei der Schwelle 0.5 fällt auf **0.39** (F1 0.56; mit bekanntem Anteil 0.84): die Scores der Anomalien rücken an 0.5 heran |
| Anomalien in der Lücke | ❌ zwei Betriebsarten, 10 % Anomalien dazwischen: AUC **0.54** (klassisch und robust 0.40), F1 0.02 – die Anomalien sind als dichte Gruppe selbst ein Modus; bei 2 % verstreuten Lücken-Anomalien 0.81; drei Betriebsarten 0.27 (unter Raten) |
| Dichte Gruppe abseits | ❌ AUC 0.95 / 0.85 / 0.70 / 0.40 bei 10 / 20 / 30 / 40 %; die robuste Schätzung 1.00 / 1.00 / 0.49 / 0.39 – **besser bis 25 %**, danach kippt sie; klassisch 0.83 / 0.62 / 0.52 / 0.44. Bei 20 % F1 0.44 mit 14 % Fehlalarmen. **Kleines ψ hilft**: bei 30 % AUC 0.84 mit ψ = 16, 0.70 mit ψ = 256 |
| Geister-Regionen | ❌ Rasterpunkte im gleichen Abstand zum nächsten Datenpunkt: Punkte **außerhalb beider Wertebereiche ("Ecken")** sind um **0.06–0.08** (bei 1–2 σ Abstand) auffälliger als solche neben dem Datenbereich ("Korridore") – Bänder entlang der Achsen; in allen gültigen Aufnahmen positiv (Abstand ab 1 σ). Die Ellipse der Wurzel ist im Abstand isotrop |
| Schwelle | ⚠️ Score-Schwelle 0.45 / 0.5 / 0.55 / 0.6 / 0.65: F1 0.82 / **0.96 / 0.97** / 0.83 / 0.55. Bei 20 Touren Fehlalarmrate 0.156 (mittlerer Score der Normalen 0.435 gegen 0.376 bei 300 Touren); ψ = 16: Fehlalarmrate 0.18, F1 0.56 – die AUC bleibt 1.00. Ein falsch angenommener Anteil (½× / 2×) senkt F1 von 0.97 auf **0.67** – bei allen drei Detektoren (robust 0.90 → 0.67, klassisch 0.69 → 0.56 / 0.58) |
| Bäume | ✅ AUC schon bei 10 Bäumen 1.00; F1 an der Schwelle 0.94 / 0.95 / 0.96 bei 10 / 25 / 50 Bäumen; Streuung des F1 über Wald-Seeds 0.037 / 0.015 / 0.009 bei 10 / 50 / 200 Bäumen |
| Kleine Abstände | ✅ AUC 0.96 / 0.99 / 1.00 bei Abstand 3 / 4 / 6 (F1 0.66 / 0.85 / 0.96); robust 0.80 / 0.93 / 1.00, klassisch 0.79 / 0.88 / 0.95 |
| Viele Merkmale | ✅ AUC 1.00 bei 2 bis 30 Merkmalen, F1 0.92–0.96; klassischer Recall 0.84 → 0.21, Fehlalarmrate der robusten Schätzung 0.03 → 0.16 |
| Invarianz | Skalierung und Verschiebung ändern den Score nicht (per Test bei gleichem Wald-Seed identisch); **Drehungen schon** (achsenparallele Schnitte) |
| Rechenzeit | ✅ 0.14 s für den Wald im Standardfall (0.3 s mit der Wurzel-Schätzung), 1.3 s bei 40 Rauschmerkmalen (die MCD-Schätzung dominiert) |

## Was die Demo zeigt

1. **Isolation Forest in Aktion** (Schritt-Slider + Abspielen): **Touren** (in der Ebene der zwei größten Streuungsrichtungen, daneben zwei Rohmerkmale) → **Zufällige Schnitte** (ein Isolationsbaum als Zerlegung der Ebene – Illustration desselben Algorithmus auf den zwei Hauptrichtungen –,
   der Weg einer Sonderfahrt und einer normalen Tour, ihre Pfadlängen in allen Bäumen des echten Waldes) → **Pfadlängen** (mittlere Pfadlänge je Tour, Anomalie-Wert nach k Bäumen) → **Score und Schwelle** (Histogramm, markierte Touren) → **Ergebnis** (Kennzahlen und ROC-Kurven).
2. **Was der Isolation Forest gefunden hat – gegen die Wurzel:** AUC, Recall, Fehlalarmrate, F1 (dazu der F1 mit bekanntem Anteil), Urteil (Codes: Lücke → dichte Gruppe → Wurzel besser → Isolation Forest besser, mit Hinweis auf eine schlechte Schwelle → falsche Schwelle bei guter Rangfolge → gleichauf), Detailtabelle.
3. **📐 Sweeps** über Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil und Abstand der Anomalien, Bäume, Unterstichprobe, Score-Schwelle, χ²-Quantil und angenommenen Anteil (feste Datensätze ab 100000, Streuung).
4. **🔬 Wald-Parameter** (Bäume × ψ, Streuung über Wald-Seeds), **🔬 Geister-Regionen** (Score-Karte über zwei Merkmale mit der Ellipse der Wurzel, Anisotropie je Betriebsart), **🔬 Wurzel-Schwächen** (Betriebsarten × Art der Anomalien), **🔬 viele Merkmale / Rauschmerkmale / wenige Touren**,
   **🔬 Schwelle** (Score-Schwelle, falscher Anteil, mittlerer Score je Tourenzahl) und **🔬 Masking** (dichte Gruppe 2–45 %, Einfluss von ψ) – Experimente auf Abruf.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (wenige verstreute Anomalien, achsenparallele Schnitte, Score als Schwelle, irrelevante Merkmale, Waldgröße, Gauß-Annahme der Wurzel).

Regler: Touren (20–600), Merkmale (2–30), Rauschmerkmale (0–40), Betriebsarten (1–3), Krümmung, Rauschen, Anteil der Anomalien (1–45 %), Art (verstreut / dichte Gruppe / in der Lücke – ab zwei Betriebsarten), Abstand (bei "Lücke" ausgeblendet, Wert bleibt erhalten),
Bäume (10–500), Unterstichprobe ψ (bis zur Tourenzahl), **Schwelle** (Standard: Score-Schwelle und χ²-Quantil, ausgeblendet beim erwarteten Anteil; sonst der angenommene Anteil für alle drei).

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | AUC IF | F1 IF | AUC robust | AUC klassisch | Urteil |
|---|---|---|---|---|---|
| Standardfall | 1.00 | 0.98 | 1.00 | 0.95 | gleich gut |
| Viele Rauschmerkmale (40) | 0.99 | 0.42 | 0.90 | 0.78 | Isolation Forest besser (Schwelle passt nicht) |
| Wenige Touren, viele Merkmale (20 × 30) | 1.00 | 0.57 | 0.64 | 0.61 | Isolation Forest besser (Schwelle passt nicht) |
| Dichte Gruppe abseits (20 %) | 0.91 | 0.64 | 1.00 | 0.56 | dichte Gruppe: Wurzel besser |
| Anomalien in der Lücke | 0.63 | 0.05 | 0.35 | 0.35 | Anomalien in der Lücke |
| Falscher erwarteter Anteil (5 % statt 10 %) | 1.00 | 0.67 | 1.00 | 0.95 | Schwelle passt nicht |

## Modell und Verfahren

- **Szenario** (`isf_scenario.py`): wortgleich aus der Wurzel (zwei versteckte Faktoren, 12 Kennzahlen der PCA-Demo, Betriebsarten, Krümmung, Anomalien verstreut / dichte Gruppe / in der Lücke mit exaktem Anteil, Zusatzmerkmale bis p = 30). Neu: **Rauschmerkmale** – unabhängige Spalten (Mittel 50, Streuung 10),
  angehängt und als letzte Ziehung, damit alle bisherigen Spalten unverändert bleiben (per Test).
- **Isolation Forest** (`isf_algorithm.py`, numpy von Grund auf): Unterstichprobe ψ ohne Zurücklegen; Baum Ebene für Ebene mit zufälligem nicht konstantem Merkmal und gleichverteiltem Schnittpunkt zwischen Minimum und Maximum des Knotens, Höhenlimit ⌈log₂ ψ⌉; Pfadlänge = Tiefe + c(Blattgröße) mit c(n) = 2 (ln(n−1) + γ) − 2(n−1)/n;
  Anomalie-Wert s = 2^(−E[h]/c(ψ)); vektorisiertes Scoring; Seed des Waldes getrennt vom Seed der Aufnahme. **Schwelle:** Standard Score über 0.5 (Faustregel) oder der **angenommene Anteil** (die größten Werte), dann für alle drei Detektoren.
- **Vergleich** (`isf_ee_algorithm.py`): klassisch und FastMCD wortgleich aus der Wurzel (χ²-Verteilung ohne scipy, Mahalanobis-Abstand über die Korrelationsmatrix, MCD mit 500 Starts, Konsistenz-Korrektur, Neugewichtung), χ²-Quantil 0.975.
- **Auswertung** (`isf_evaluation.py`): AUC, mittlere Präzision, Precision, Recall, F1, Fehlalarmrate; **F1 mit bekanntem Anteil** als Referenz für die Schwelle; **Score-Anisotropie** (Ecken minus Korridore bei gleichem standardisiertem Abstand zum nächsten Datenpunkt ±0.15, Wald über die ersten zwei Merkmale);
  Sweeps, Experiment-Tabellen, Urteil.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutungen, die nicht stimmten:** (1) "Rauschmerkmale senken die AUC des Isolation Forest merklich" – gemessen bleibt sie bei 0.99 (die robuste Schätzung fällt auf 0.88); was leidet, ist die **Schwelle** (Recall 0.39). (2) "Auf gekrümmtem Normalbereich bleibt der Isolation Forest gut, die Wurzel nicht" – die **Rangfolge** ist für alle drei gleich (AUC 1.00);
  der Unterschied entstand nur durch die χ²-Schwelle der Wurzel, nicht durch den Detektor. (3) "Im Gauß-Fall ist die MCD-Rangfolge mindestens so gut" – sie ist gleich (1.00), der Isolation Forest hat dank der Schwelle das bessere F1. (4) "Der Isolation Forest findet Lücken-Anomalien, die die Wurzel nie findet" – nur bei wenigen verstreuten (2 %: 0.81);
  bei 10 % sind sie eine dichte Gruppe und AUC 0.54 ist kaum über Raten.
- **Der Score ist keine kalibrierte Wahrscheinlichkeit:** die Schwelle 0.5 stimmt nur für große Stichproben (Score der Normalen 0.376 bei 300 Touren, 0.435 bei 20) und große ψ (ψ = 16: Fehlalarmrate 0.18); bei vielen Rauschmerkmalen ist sie zu hoch. Die **Rangfolge** bleibt in all diesen Fällen gut – der Isolation Forest ist ein Detektor ohne eingebaute Entscheidung.
- **Der angenommene Anteil ist Vorwissen:** die Standardeinstellung des Anteils entspricht dem wahren; bei ½× oder 2× fällt F1 von 0.97 auf 0.67 – aber bei allen drei Detektoren gleich, denn dann entscheidet nur die Rangfolge.
- **Masking:** eine dichte Gruppe abseits ist auch in der Unterstichprobe dicht (AUC 0.85 bei 20 %), die robuste Schätzung ist bis 25 % besser; bei 30 % kippt sie (0.49), der Isolation Forest bleibt über Raten (0.70). Ab 40 % versagen alle: die Gruppe *ist* der Normalbereich.
- **Geister-Regionen und Drehungen:** achsenparallele Schnitte bewerten Punkte neben dem Datenbereich anders als solche in den Ecken; das ist die Schwäche, an der der Extended Isolation Forest ansetzt (noch nicht gebaut).
- **Synthetische Daten:** zwei Faktoren, lineare Mischung, weißes Gauß'sches Rauschen, feste Betriebsarten-Geometrie; die Anomalien liegen im Faktorraum weit draußen (Abstand 3–12 σ), was allen Verfahren entgegenkommt. Literatur nur mit Namen: Liu, Ting und Zhou (Isolation Forest); Hariri, Kind und Brunner (Extended Isolation Forest).

## Verifikation

- Isolation Forest: c(n) gegen die Definition und die exakte harmonische Summe; Baumstruktur (Knotengrößen = Summe der Kinder, Blattgrößen = wirklich dort landende Punkte, Schnittpunkt im Wertebereich des Knotens, Tiefe ≤ Limit, konstante Merkmale nie geschnitten, gleiche Punkte in einem Blatt);
  Pfadlänge = expliziter Weg; **Score gegen `sklearn.ensemble.IsolationForest`** (Rangkorrelation > 0.9, AUC-Übereinstimmung < 0.02, gleiche Skala); Skalen- und Verschiebungsinvarianz (identische Scores), keine Drehungsinvarianz; Determinismus und Seed-Abhängigkeit; Konvergenz mit mehr Bäumen; Score 0.5 bei mittlerer Pfadlänge c(ψ).
- Wurzel-Schätzer: χ² gegen scipy, MCD gegen `sklearn.covariance.MinCovDet`, affine Äquivarianz. Szenario: normale Zeilen wie in der PCA-Demo (eingefrorene Zeilensummen), eingefrorener Standardfall, `n_noise` ändert nur angehängte Spalten, exakter Anomalie-Anteil, Geometrie der Arten.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Bruchpunkt-/Masking-/Betriebsarten-/Schwellen-/Dimensions-/Geister-Tabellen; jeweils Mittel über die festen Sweep-Datensätze, positive **und** negative Aussagen);
  alle 6 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt bei 2, 12 und 30 Merkmalen, wenige Touren mit vielen Merkmalen und Rauschmerkmalen, ψ-Grenze folgt der Tourenzahl, ausgeblendete Schwellenregler behalten ihre Werte, Sweep-Optionen folgen der Schwellenart, Art "Lücke" nur ab zwei Betriebsarten,
  Experimente auf Abruf), Achsensperre und explizite eindeutige Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Parameter, Geister, Wurzel-Schwächen, Dimension, Schwelle, Masking), 🚧 Grenzen, Mathe |
| `isf_algorithm.py` | Isolationsbaum, Wald, Pfadlängen, Score, Schwelle, Baum-Zerlegung für die Darstellung |
| `isf_ee_algorithm.py` | χ²-Verteilung, klassische Schätzung, FastMCD (wortgleich aus elliptic-envelope-demo) |
| `isf_scenario.py`, `isf_constants.py` | Touren mit Betriebsarten, Krümmung, Anomalien und Rauschmerkmalen; Konstanten, Presets |
| `isf_evaluation.py` | Kennzahlen, Analyse, Schwellen, Sweeps, Experimente, Anisotropie, Urteil |
| `isf_presets.py`, `isf_visualization.py` | Permalink/Presets (ausgeblendete Regler, ψ-Grenze), Plotly-Figuren (achsengesperrt) |
| `tests/` | Wald (sklearn-Kreuzprüfung, Handinstanzen, Invarianzen), Szenario und Kennzahlen, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
