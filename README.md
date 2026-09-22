# NegativeKANNeXt

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Research code](https://img.shields.io/badge/status-research%20code-lightgrey.svg)](#scope-and-clinical-use)

MATLAB implementation of **NegativeKANNeXt**, a compact subtraction-based convolutional neural network, together with the associated deep feature engineering (DFE) pipeline for anterior cruciate ligament (ACL) rupture classification from sagittal knee MRI.

This repository accompanies the manuscript:

> **A Novel Negative-Based CNN and Deep Feature Engineering Framework for ACL Rupture Detection from Knee MRI**

**Authors:** Sukru Demir, Bugra Can, Omer Faruk Goktas, Mehmet Baygin, Sengul Dogan, and Turker Tuncer.

Repository: https://github.com/mhmtbygn/NegativeKANNeXt

## Method overview

The repository implements the principal computational workflow reported in the manuscript:

- NegativeKANNeXt with Negative-Stem, NegativeKAN, and Negative-Down modules;
- subtraction-based feature fusion;
- dual GELU/Swish activation paths;
- patient-level splitting before augmentation;
- training-only augmentation;
- end-to-end classification;
- extraction of the 768-dimensional global-average-pooling representation;
- CWINCA-based feature selection;
- shallow classification using kNN, cubic SVM, decision tree, linear discriminant, kernel Naive Bayes, and logistic regression;
- held-out patient-level evaluation;
- repeated patient-level cross-validation;
- calibration utilities;
- ablation utilities;
- external-domain evaluation entry point;
- Grad-CAM and occlusion-sensitivity utilities;
- representation-analysis utilities.

The implementation intentionally separates training, evaluation, feature selection, and held-out testing so that test observations do not participate in model fitting or feature selection.

## Reported study results

The associated manuscript reports the following principal results:

| Evaluation | NegativeKANNeXt | DFE + SVM |
|---|---:|---:|
| Repeated patient-level cross-validation accuracy | 93.54 ± 2.32% | 95.75 ± 2.32% |
| Held-out, non-augmented test accuracy | 93.38% | 96.32% |
| Frozen-model external validation accuracy | 83.33% | — |

The lightweight two-class NegativeKANNeXt configuration is reported as approximately **6.4 million parameters** and **2.03 GFLOPs**.

These values are manuscript results. Re-running the code can produce small numerical differences because of software, hardware, random-state, and implementation-level effects.

## Repository structure

```text
NegativeKANNeXt/
├── README.md
├── ASSUMPTIONS_AND_SCOPE.md
├── CITATION.cff
├── CHANGELOG.md
├── LICENSE
├── VERSION
├── run_all.m
├── config/
│   └── defaultConfig.m
├── data/
│   ├── prepare_dataset.m
│   └── augment_training_set.m
├── model/
│   └── buildNegativeKANNeXt.m
├── training/
│   └── train_holdout.m
├── dfe/
│   ├── cwinca_select.m
│   ├── apply_cwinca.m
│   ├── train_shallow_classifiers.m
│   └── run_dfe_pipeline.m
├── evaluation/
│   ├── classification_metrics.m
│   ├── calibration_analysis.m
│   └── evaluate_network.m
├── cv/
│   └── run_repeated_nested_cv.m
├── ablation/
│   └── run_ablation.m
├── external/
│   └── run_external_validation.m
├── xai/
│   └── run_xai.m
├── analysis/
│   ├── parameter_audit.m
│   └── representation_analysis.m
├── utils/
│   ├── export_environment.m
│   └── wilson_ci.m
└── tests/
    └── smoke_test.m
```

## Requirements

The code was prepared for **MATLAB R2023b or later** and uses functionality from:

- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox
- Image Processing Toolbox
- Parallel Computing Toolbox (recommended for GPU execution)

The manuscript experiments were performed in MATLAB R2023b.

## Dataset organization

The source MRI data are **not redistributed in this GitHub repository**. Users should place an authorized copy of the single-slice dataset in the following form:

```text
data_original/
├── ACL_Rupture/
│   ├── patient_0001.png
│   └── ...
└── Control/
    ├── patient_0002.png
    └── ...
```

The code assumes one representative sagittal slice per patient.

The manuscript cohort contains:

- 317 ACL-rupture patients;
- 364 controls;
- 681 patients in total.

For the principal hold-out experiment, the code reproduces the reported stratified patient-level split:

| Class | Training | Validation | Test |
|---|---:|---:|---:|
| ACL rupture | 222 | 32 | 63 |
| Control | 255 | 36 | 73 |
| **Total** | **477** | **68** | **136** |

Partitioning is performed **before augmentation**. Only the training partition is augmented.

## Quick start

Clone the repository:

```bash
git clone https://github.com/mhmtbygn/NegativeKANNeXt.git
cd NegativeKANNeXt
```

Open MATLAB in the repository root and run:

```matlab
addpath(genpath(pwd));
cfg = defaultConfig();
```

Set the local dataset path in `config/defaultConfig.m`:

```matlab
cfg.DataRoot = fullfile(pwd,"data_original");
```

Run the principal hold-out workflow:

```matlab
run_all
```

Or execute the stages individually:

```matlab
cfg = defaultConfig();

split = prepare_dataset(cfg);
augTrain = augment_training_set(split.train,cfg);

[net,trainInfo] = train_holdout(augTrain,split.val,cfg);

holdout = evaluate_network(net,split.test,cfg);

dfe = run_dfe_pipeline(net,augTrain,split.test,cfg);
```

## Repeated patient-level cross-validation

The repeated validation protocol can be run with:

```matlab
cfg = defaultConfig();
results = run_repeated_nested_cv(cfg);
```

The default configuration uses five repetitions of stratified five-fold patient-level cross-validation.

This procedure is computationally expensive because the network is reinitialized and retrained for each outer fold.

## Parameter audit

The analytic parameter-count utility can be run with:

```matlab
cfg = defaultConfig();
parameter_audit(cfg);
```

Under the default maximal-grouping interpretation, the implementation contains approximately **6.364 million trainable parameters**, consistent with the manuscript's rounded **6.4 million** value.

## Deep feature engineering

The DFE workflow uses the global-average-pooling representation from NegativeKANNeXt.

The manuscript configuration uses:

- 768 GAP features;
- CWINCA lower cumulative-weight threshold: 0.05;
- CWINCA upper cumulative-weight threshold: 0.99;
- 1-NN loss estimator;
- Cityblock (L1) distance;
- equal distance weighting;
- 10-fold cross-validation within the training data.

The reported held-out configuration retained four GAP features:

```text
48, 571, 141, 338
```

By default, the repository recomputes feature selection from the supplied training data. To force the manuscript-reported feature indices for a compatible trained model, set:

```matlab
cfg.DFE.UseReportedIndices = true;
```

## External validation

`external/run_external_validation.m` implements frozen-model evaluation on an independently prepared external set.

Because the manuscript does not fully specify an automated label-blinded representative-slice selection algorithm for MRNet, this repository does **not** fabricate one. External representative sagittal slices must therefore be prepared independently of ACL labels before the evaluation function is called.

## Explainability

The repository includes reproducible MATLAB utilities for:

- Grad-CAM;
- occlusion sensitivity;
- optional comparison with expert masks when such annotations are available.

The manuscript also reports Score-CAM, Integrated Gradients, LIME, and feature-level Shapley analyses. Their reported parameter settings are preserved in `config/defaultConfig.m` and `ASSUMPTIONS_AND_SCOPE.md`. Exact reproduction of expert-mask-based quantitative XAI requires the corresponding expert annotations.

## Reproducibility notes

The manuscript determines the main architecture and experimental protocol, but a small number of implementation-level details are not uniquely specified in the article text. These points are documented explicitly in:

[`ASSUMPTIONS_AND_SCOPE.md`](ASSUMPTIONS_AND_SCOPE.md)

No unspecified detail is presented as experimentally verified historical source code.

## Pretrained model

The trained MATLAB network is not included in the Git source tree. A trained `.mat` model can be distributed as a versioned release asset or archived together with the code in Zenodo. Keeping large binary model files outside the ordinary Git history makes the source repository easier to clone and maintain.

## Scope and clinical use

This repository is provided for **research and reproducibility purposes**.

The associated study evaluates a single-slice ACL classification framework. It is not an autonomous clinical diagnostic system, and the reported results should not be interpreted as complete MRI-examination diagnostic performance. Clinical use would require evaluation on complete MRI series and prospective multi-center validation across scanners, acquisition protocols, and heterogeneous ACL injury patterns.

## Citation

GitHub can generate a software citation directly from [`CITATION.cff`](CITATION.cff).

Until a final article DOI and archived software DOI are available, cite the repository as:

> Demir S, Can B, Goktas OF, Baygin M, Dogan S, Tuncer T. **NegativeKANNeXt reproducibility code**. Version 1.0.0, 2026. https://github.com/mhmtbygn/NegativeKANNeXt

When the manuscript and Zenodo archive receive permanent DOIs, those identifiers can be added to `CITATION.cff` without changing the source-code version.

## License

The source code in this repository is released under the [MIT License](LICENSE).

The license applies to the software only. It does not grant rights to any clinical MRI data, third-party datasets, or third-party software.

## Contact

For questions regarding the software or associated study, please use the GitHub Issues page or contact the authors through the affiliations reported in the manuscript.
