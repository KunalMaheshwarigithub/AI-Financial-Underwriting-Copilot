# Model documentation

## Objective

The model estimates the probability of the historical target outcome in the Home Credit Default Risk dataset. In this project, the saved estimator is a `HistGradientBoostingClassifier` with a preprocessing pipeline.

## Dataset

Source: [Home Credit Default Risk — Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk)

Recorded `application_train` cohort:

| Item | Count | Share |
|---|---:|---:|
| Total applications | 307,511 | 100.00% |
| `TARGET = 0` | 282,686 | ~91.93% |
| `TARGET = 1` | 24,825 | ~8.07% |

The target is a historical outcome label, not an input feature. The applicant identifier `SK_ID_CURR` is also excluded from model predictors.

## Feature schema

- Total configured predictors: **244**
- Numeric predictors: **228**
- Categorical predictors: **16**
- The API reconstructs the input DataFrame in the exact feature order stored in `models/feature_config.json`.
- Missing values are passed through the saved preprocessing pipeline according to the training-time imputation strategy.

Do not create synthetic bureau, external credit-source, or repayment values merely to fill missing columns. For new applications, unavailable history must remain missing and must be explicitly disclosed in the assessment.

## Estimator and preprocessing

- Estimator: `HistGradientBoostingClassifier`
- Preprocessing: saved scikit-learn transformer loaded from the model artifacts.
- Persisted artifacts: model, preprocessor, feature configuration, model metadata, policy configuration, and risk-band configuration.
- Model version recorded in project context: `historical_features_v1`.

The exact preprocessing parameters should be taken from the serialized artifact and the single `AI_Financial_Project` Kaggle notebook rather than inferred from this summary.

## Recorded evaluation metrics

| Metric | Recorded value |
|---|---:|
| ROC-AUC | 0.782369 |
| PR-AUC | 0.278541 |
| Brier score | 0.066097 |

Interpretation:

- ROC-AUC measures ranking discrimination across thresholds.
- PR-AUC summarizes precision-recall trade-offs and is useful when the positive class is uncommon.
- Brier score measures the mean squared error of probabilistic predictions; lower is better.

These values are the metrics recorded in the project's previous model evaluation. Before treating them as a reproducible benchmark, the `AI_Financial_Project` Kaggle notebook should document the exact train/validation split, random seeds, preprocessing fit boundaries, metric implementation, and calibration analysis.

## Risk bands and project policy thresholds

The project currently stores risk bands and deterministic policy thresholds in JSON configuration files. They are prototype assumptions for demonstrating software workflow. They are not official bank policies, regulatory thresholds, or evidence that a lending decision is appropriate.

Unavailable evidence should be represented as `NOT_EVALUATED`. It must not be silently converted into a policy pass or policy failure.

## New-applicant limitation

The historical estimator expects 244 fields, while the current new-application form supplies only a subset. Median/constant imputation can make the input technically processable, but the resulting score can be affected by a large amount of missing historical information and distribution shift.

A score returned for a new applicant must therefore be described as an experimental model output, not as a validated prediction for that applicant population. A responsible next step is to evaluate feature coverage, compare input distributions with training data, assess calibration on an appropriate held-out cohort, and validate performance for the intended application population.

## Fuzzy and historical-comparison components

Fuzzy memberships are explanatory indicators for supported inputs. They are not probabilities of default unless separately trained and calibrated as such.

Historical cohort comparisons can provide context only when the reference data are suitable, labels are available, and the cohort size and similarity are reported. The cohort observed default rate is descriptive and must not be presented as an individual causal prediction.

## Responsible-use checklist

- [ ] Reproduce the model metrics from versioned notebooks.
- [ ] Confirm no target leakage or applicant-ID leakage.
- [ ] Confirm preprocessing is fitted only on training folds.
- [ ] Add calibration curves and threshold analysis.
- [ ] Evaluate missingness and out-of-distribution inputs.
- [ ] Assess fairness and subgroup performance using a legally and ethically appropriate protocol.
- [ ] Validate fuzzy rules and similarity methodology independently.
- [ ] Establish human review and governance before any real-world use.
