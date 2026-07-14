# Movie Success Prediction (Classification)

Predicting whether a movie earns a technical award nomination (`Start_Tech_Oscar`) using pre-release and production attributes, built independently to apply classification techniques learned during coursework.

## Overview

This project explores a movie dataset covering budget, marketing spend, cast/crew ratings, and social media buzz, then builds and compares several classification models to predict award nomination likelihood.

## Tech Stack

- **Python**
- **Pandas, NumPy** — data cleaning and manipulation
- **Seaborn, Matplotlib** — exploratory visualization
- **Scikit-learn** — Logistic Regression, Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), GridSearchCV
- **Statsmodels** — logistic regression significance testing

## Workflow

### 1. Exploratory Data Analysis
- Used boxplots and joint plots to inspect distributions of `Time_taken`, `Collection`, `Twitter_hastags`, `Marketing expense`, and `Production expense`.
- Identified outliers in `Avg_age_actors`, `Twitter_hastags`, `Collection`, and `Marketing expense`.
- Identified missing values in `Time_taken`.
- Found `MPAA_film_rating` added no useful signal for prediction.

### 2. Data Cleaning & Feature Engineering
- Capped outliers in `Twitter_hastags` (above the 99th percentile) and `Avg_age_actors` (below the 1st percentile).
- Filled missing values in `Time_taken` with the column mean.
- Combined four separate rating columns (`Lead_Actress_rating`, `Director_rating`, `Producer_rating`, `Critic_rating`) into a single `Avg_rating` feature.
- One-hot encoded categorical variables (e.g. `3D_available`, `Genre`) using `pd.get_dummies`.

### 3. Modeling
- **Logistic Regression** — first tested with a single predictor (`Budget`) to check statistical significance via `statsmodels`, then extended to a multi-predictor model using all available features.
- **Linear Discriminant Analysis (LDA)** — trained on the full feature set for comparison against logistic regression.
- **K-Nearest Neighbors (KNN)** — features scaled with `StandardScaler`, tested with k=1 and k=3, then tuned across k=1–10 using `GridSearchCV`.

### 4. Evaluation
Model performance evaluated with confusion matrices and accuracy scores on a held-out test set (80/20 split):

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 63.7% |
| KNN (k=1) | 57.8% |
| KNN (k=3) | 54.9% |

Logistic Regression performed best on the held-out test set among the models compared.

## Key Learnings

- Single-predictor logistic regression (`Budget` alone) was not statistically significant (p > 0.05), showing the importance of testing predictor significance rather than assuming a variable matters.
- Outlier capping and missing value imputation meaningfully affect model input quality — applied before any model training.
- Comparing multiple classification algorithms on the same data highlights trade-offs; simpler logistic regression outperformed KNN on this dataset.

## Possible Extensions

- Cross-validation instead of a single train/test split for more robust accuracy estimates.
- Feature scaling review — currently `StandardScaler` is fit separately on train and test sets; fitting once on train and applying to test would be more standard practice.
- Precision/recall/F1 alongside accuracy, since class balance affects how meaningful accuracy alone is.

## Dataset

Movie dataset (`Movie collection.csv`) containing production, marketing, cast/crew rating, and social media metrics for a set of films.
