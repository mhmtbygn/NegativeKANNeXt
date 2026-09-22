# NegativeKANNeXt reproducibility package (v1.0)

This repository is a clean MATLAB R2023b-oriented implementation of the methods described in:

**A Novel Negative-Based CNN and Deep Feature Engineering Framework for ACL Rupture Detection from Knee MRI**

Authors: Sukru Demir, Bugra Can, Omer Faruk Goktas, Mehmet Baygin, Sengul Dogan, Turker Tuncer.

## What is included

The package contains:

- patient-level 70:10:20 stratified split with seed 42;
- deterministic generation of 25 augmented training variants per patient;
- NegativeKANNeXt architecture:
  - Negative-Stem,
  - four NegativeKAN blocks,
  - three Negative-Down blocks,
  - GAP + FC + Softmax output;
- SGDM training configuration reported in the manuscript;
- held-out patient-level evaluation;
- GAP feature extraction (768 features);
- CWINCA-style NCA-guided feature selection;
- shallow classifiers (kNN, cubic SVM, DT, LD, NB, LR);
- calibration analysis;
- repeated patient-level cross-validation / nested DFE evaluation;
- configurable ablation variants;
- external-domain evaluation entry point;
- XAI utilities and representation-analysis utilities;
- an assumptions/reproducibility note documenting details that are not fully determined by the manuscript text.

## MATLAB requirements

Designed for MATLAB R2023b or later with:

- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox
- Image Processing Toolbox
- Parallel Computing Toolbox (recommended for GPU execution)

## Expected dataset layout

Place the original, non-augmented representative sagittal slices in:

```text
DATA_ROOT/
  ACL_Rupture/
    patient_0001.png
    ...
  Control/
    patient_0002.png
    ...
```

Exactly one image per patient is expected. The manuscript cohort contains:

- ACL rupture: 317 patients
- Control: 364 patients
- Total: 681 patients

The split produced by `prepare_dataset` is fixed to:

- training: 222 ACL + 255 Control = 477 patients
- validation: 32 ACL + 36 Control = 68 patients
- test: 63 ACL + 73 Control = 136 patients

## Quick start

1. Edit `config/defaultConfig.m` and set `cfg.DataRoot`.
2. In MATLAB, add the repository recursively:

```matlab
addpath(genpath(pwd));
```

3. Run:

```matlab
run_all
```

For a smaller staged workflow:

```matlab
cfg = defaultConfig();
split = prepare_dataset(cfg);
augTrain = augment_training_set(split.train, cfg);
[net, info] = train_holdout(augTrain, split.val, cfg);
holdout = evaluate_network(net, split.test, cfg);
dfe = run_dfe_pipeline(net, augTrain, split.test, cfg);
```

## Reproducibility principles

- Patient-level splitting occurs before augmentation.
- Only the training partition is augmented.
- Validation and test images remain original and non-augmented.
- Feature selection is fit on training data only.
- Selected feature indices are frozen before held-out testing.
- The test partition is not used for model selection.

## Important implementation note

The manuscript specifies the mathematical block operations but does not explicitly report every grouped-convolution group count. The default implementation uses the **maximum valid grouping** (`gcd(inputChannels, outputChannels)`) for grouped convolutions. For Negative-Down, grouped stride-2 convolutions are enabled by default because this interpretation is consistent with the reported compact parameter scale (~6.4 M). Set:

```matlab
cfg.Model.UseGroupedDownsample = false;
```

to use standard convolutions in Negative-Down exactly as the equation symbol `C(.)` may be read.

The corresponding analytic parameter audit gives approximately **6.364 million trainable parameters**, which rounds to the **6.4 million** reported in the manuscript. Run:

```matlab
cfg = defaultConfig();
parameter_audit(cfg)
```

See `ASSUMPTIONS_AND_SCOPE.md` before public release.

## Code availability statement template

> The custom MATLAB code used to implement NegativeKANNeXt, the deep feature engineering pipeline, CWINCA-based feature selection, model evaluation, and the principal analytical procedures reported in this study is publicly available in the archived repository associated with this article. The archived version corresponds to the implementation used to generate the reported analyses. DOI: [TO BE ADDED AFTER ZENODO ARCHIVING].

## Release checklist

Before uploading to GitHub/Zenodo:

1. run `tests/smoke_test.m`;
2. run the complete pipeline on the final dataset;
3. compare layer/parameter counts with the manuscript;
4. verify the final selected CWINCA indices;
5. replace placeholders in `CITATION.cff`;
6. select a software license;
7. create a Zenodo release and insert its DOI into the manuscript.
