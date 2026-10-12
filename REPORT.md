# Assignment 1 Report

- **Name**: Jaime Lopez Baillo
- **Student ID**: [your student ID]
- **Email**: jlopezbaillo.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

---

## Dataset

**What it is:** Lap-by-lap telemetry from Formula 1 races (2022–2025), one row per driver per lap. Each row captures the tyre compound fitted, how many laps those tyres have run (`TyreLife`), cumulative time lost to degradation, current race position, and whether the car made a pit stop on that lap (`PitStop`) or will pit on the *next* lap (`PitNextLap`).

**Source:** Provided as `train (2).csv` and `test (2).csv` (no external download).

**Size:** 439,140 training rows × 16 columns. 104 unique race-year events spanning 20 drivers.

**Why I picked it:** Pit stop timing is the single biggest lever in F1 strategy. The dataset has a clear binary target, naturally rich features, and a realistic business question that drives every pipeline decision — making it much more interesting than a generic tabular benchmark.

---

## Business / real-life framing

**Scenario:** A race engineer on an F1 team must decide each lap whether to call their *own* driver in for a pit stop, partly in response to what *rival* teams are doing. If a rival pits and you do not react, they emerge on fresh tyres, set faster laps, and undercut you — costing track position worth tens of thousands of euros in constructors' points prize money.

**The model's role:** Predict whether a rival car will pit on the *next* lap, one lap in advance, so the strategy desk can prepare a counter-pit response.

**Pipeline implications this drives:**

1. **Target definition:** `PitNextLap` is already binary — 0 = no pit next lap, 1 = will pit. No engineering needed, but the timing matters: we are predicting one lap ahead, which is exactly the window an engineer needs to act.

2. **Split strategy — race-based, not random:** A random row-level split would put, for example, lap 50 of race X in test while lap 30 of the same race is in training. That leaks race-specific context (safety cars, strategy phase, tyre-life trajectory) into the model during evaluation in a way that would not occur in real deployment, where you only see the current lap when making the call. I therefore split on whole races: each (Race, Year) group appears entirely in train, validation, or test — never split across sets.

3. **Metric:** AUC-ROC summarises performance across all thresholds and is the right primary metric here because the optimal threshold depends on the cost trade-off (missed undercut vs. unnecessary alert), which the team sets externally. F1 at threshold 0.5 is reported for comparability. A missed undercut (false negative) is far more expensive than a false alarm (false positive), so at deployment the threshold should be tuned down from 0.5 — the dashboard's threshold tool allows this interactively.

---

## Data preparation & feature engineering

**No missing values** — the dataset arrived clean.

**Compound encoding:** The five compounds (WET, INTERMEDIATE, HARD, MEDIUM, SOFT) were mapped to an ordinal softness score (0–4) based on their degradation rate. One-hot indicators for SOFT, MEDIUM, and HARD capture non-linear compound effects without assuming a strict linear order.

**Tyre wear ratio:** `TyreLife / expected_stint_length` normalises tyre age against the compound's typical lifespan (HARD≈45 laps, MEDIUM≈35, SOFT≈25, INTERMEDIATE≈25, WET≈20). A SOFT tyre with 20 laps is far more worn than a HARD tyre with 20 laps; the raw age alone misses this.

**Quadratic tyre age (`tyre_life_sq`):** Degradation in F1 is non-linear — the last few laps on a worn set are dramatically slower. Squaring TyreLife lets the linear model approximate this curve.

**Interaction (`compound_code × TyreLife`):** Soft tyres degrade faster per lap, so the slope of time-loss vs. age differs by compound. This term lets a logistic regression capture that interaction.

**Race remaining (`1 − RaceProgress`):** Teams are more likely to pit when there are enough laps left to benefit from fresh tyres; they stretch stints at the end. This is distinct from `RaceProgress` and both are included.

**`PitStop` (current lap):** A car that pitted this lap will almost certainly not pit again next lap, so this is a strong negative signal.

**Features excluded:** Driver ID and Race name are categorical with many levels. Including them as dummies would bloat the model and risk encoding track-specific patterns that hurt generalisation across new races. They are dropped. `LapTime (s)` is excluded because the delta and cumulative degradation already capture pace-relative information; the raw lap time varies by circuit length and adds noise.

**Scaling:** StandardScaler (zero mean, unit variance) fitted on the training races only, applied identically to validation and test.

---

## Modeling: three implementations, one model

**Why logistic regression:** The target is binary. Logistic regression is the natural choice — interpretable coefficients, well-understood regularisation, and directly comparable across sklearn and PyTorch.

**Class imbalance handling:** 20% of rows have `PitNextLap=1`. This is moderate imbalance. sklearn uses `class_weight='balanced'` (re-weights the loss by inverse class frequency). PyTorch uses `pos_weight = n_negatives / n_positives ≈ 3.87` in `BCEWithLogitsLoss` and in the manual loop's weighted BCE formula.

### Three-method comparison

| Method | Accuracy | F1 | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|
| scikit-learn | 0.7607 | 0.5223 | **0.8432** | 0.3899 | 0.7911 |
| Manual PyTorch | 0.7548 | 0.5164 | 0.8364 | 0.3832 | 0.7915 |
| PyTorch nn.Module | 0.7494 | 0.5118 | 0.8331 | 0.3775 | 0.7942 |
| Naive baseline | 0.8346 | 0.0000 | 0.5000 | — | 0.0000 |

*All three trained on the same 73-race training set (301,195 rows), evaluated on the same 16-race test set (76,757 rows). Threshold = 0.5 for accuracy, F1, precision, recall; AUC is threshold-independent.*

**Do the three agree?** Yes, closely. AUC spans only 0.010 across methods. The ordering (sklearn > manual PyTorch > PyTorch nn.Module) reflects convergence differences, not fundamental disagreement: sklearn's L-BFGS solver converges to a tighter optimum in fewer equivalent gradient evaluations than SGD-family methods with the same epoch budget. All three are substantially better than the naive baseline on the metrics that matter (F1, AUC).

**Why accuracy is misleading:** The naive "always predict no pit" baseline scores 83.5% accuracy — the best of any method — because it simply predicts the majority class. This is why AUC and F1 are the real evaluation criteria here.

**Manual PyTorch implementation notes:** Forward pass: `sigmoid(X @ w + b)`. Loss: manually coded weighted BCE. Update: `w -= lr * w.grad; w.grad.zero_()` inside a `torch.no_grad()` block. No optimizer object, no nn.Module.

---

## Limitations & next steps

1. **No circuit features:** Each track has different pit stop windows (Monaco has very few, Bahrain has many). Including circuit-length-normalised lap counts or a track ID embedding would likely improve AUC by 2–3 points. A fixed embedding per circuit could be added without overfitting risk if regularised.

2. **Race dynamics ignored:** Safety car periods trigger mass pit stops, creating label noise — many cars pit next lap not because their tyres are worn but because of an opportunistic window. The model has no signal for safety-car probability. Adding a rolling variance in lap times (a proxy for pack disruption) would partially address this.

3. **Single-lap lookahead only:** Teams sometimes make strategic decisions two or three laps in advance. Extending the target to "pits within the next N laps" (with N selectable) would make the model more actionable for pre-planning.

4. **Test set positive rate discrepancy:** The training set has 20.5% positives while the test set has only 16.5%. This happens because the race-based split is not stratified on positive rate — some races (e.g., wet races with many pit stops) landed in training by chance. In practice I would stratify by positive rate across splits, or use a time-ordered split (train on 2022–2023, test on 2024–2025) for a more realistic evaluation of generalization to future races.

---

## Generative AI use disclosure

Claude Code (claude-sonnet-4-6) was used to write the boilerplate for this assignment: the training loop scaffolding, Gradio layout code, and plotly chart builders. All decisions about business framing, feature engineering (what to include, how to encode it, and why), split strategy, model configuration, and the analysis in this report are my own. The AI was used as a coding accelerator, not as a substitute for understanding the material.
