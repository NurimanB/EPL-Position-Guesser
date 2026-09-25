# EPL Position Guesser

Predicts a Premier League player's position — **Defender**, **Midfielder**, or **Forward** — from their 2019/20 season statistics.

Goalkeepers are excluded (their stat sheet is a different shape entirely), and the outfield stats are turned into **per-match rates** rather than raw season totals, so a 35-appearance regular and a 5-appearance squad player are compared fairly.

## Results

Two classifiers trained on the same engineered features:

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 88% | strong linear baseline |
| Random Forest (100 trees) | **89%** | best per-class F1 on Defender (0.97) |

```
              precision    recall  f1-score   support

    Defender       0.97      0.97      0.97        32
     Forward       0.94      0.70      0.80        23
  Midfielder       0.81      0.94      0.87        36
```

**Where it struggles:** Midfielder vs. Defender. Modern defensive midfielders rack up tackles and interceptions at rates close to actual defenders, so the model leans on goal/shot volume to tell them apart — and misses when a low-scoring winger or a tackle-heavy #6 breaks that pattern (10 of 91 test-set errors, all along this boundary).

## Approach

1. **Clean** — drop duplicates, drop goalkeepers, drop players with ≤10 appearances (too little signal, causes divide-by-zero downstream).
2. **Engineer features** — convert 13 counting stats (goals, assists, passes, tackles, cards, etc.) to per-match rates; keep `Age` as-is.
3. **Explore** — correlation heatmap and per-position boxplots to sanity-check the per-match features actually separate the classes.
4. **Train** — 75/25 stratified split, `StandardScaler` for the logistic model (trees don't need it), both models compared on the same test set.
5. **Diagnose** — confusion matrix, feature importances, and a manual look at every misclassified player.

## Data

`dataset - 2020-09-24.csv` — 2019/20 Premier League player stats (appearances, goals, passing, defensive actions, disciplinary record, etc.), 363 outfield players after cleaning.

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook positions.ipynb
```

## Stack

Python · pandas · scikit-learn (LogisticRegression, RandomForestClassifier) · seaborn/matplotlib
