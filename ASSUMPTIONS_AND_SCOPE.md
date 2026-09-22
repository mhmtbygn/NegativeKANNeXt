# Reproducibility Notes and Implementation Scope

This document distinguishes information explicitly specified in the manuscript from implementation choices required to convert the manuscript description into executable MATLAB code.

## Directly specified in the manuscript

The following settings are taken directly from the manuscript:

- MATLAB R2023b.
- Input size: 224 x 224 x 3.
- Random seed for the principal 70:10:20 patient-level split: 42.
- Cohort: 681 patients, with one representative sagittal slice per patient.
- Training / validation / test counts: 477 / 68 / 136 patients.
- Augmentation is performed only after patient-level partitioning and only on training data.
- Twenty-five augmented variants are generated per training image, producing 11,925 augmented training images.
- Optimizer: SGDM.
- Initial learning rate: 0.01.
- Momentum: 0.9.
- L2 regularization: 1e-4.
- Mini-batch size: 200.
- Maximum number of epochs: 40.
- Constant learning rate.
- Training samples reshuffled every epoch.
- Validation frequency: 50 iterations.
- No early stopping.
- Glorot weight initialization.
- Zero bias initialization.
- Zero-center input normalization.
- No class weighting.
- Main architecture widths: 96, 192, 384, and 768 channels.
- GAP feature length: 768.
- CWINCA thresholds: 0.05 and 0.99.
- CWINCA loss model: 1-NN with Cityblock/L1 distance, equal weighting, feature standardization, and 10-fold cross-validation.
- Reported held-out selected features: 48, 571, 141, and 338.
- DFE classifiers: kNN, cubic SVM, decision tree, linear discriminant, kernel Naive Bayes, and logistic regression.
- Integrated Gradients: all-zero baseline and 50 integration steps.
- LIME: 64 interpretable features and 3,072 perturbation samples.
- Score-CAM: positive feature activations, activation-range threshold of 1e-8, and batch size of 64.
- Multi-method occlusion comparison: 24 x 24 mask and stride 8.
- Detailed class-specific occlusion analysis: 16 x 16 mask and stride 8.
- Representative Figure 5 occlusion analysis: 24 x 24 mask and stride 12.

## Architecture recovered from the manuscript equations

### Negative-Stem

```text
tenNS = BN( GELU(C7x7,s4,96(Img)) - C4x4,s4,96(Img) )
```

### NegativeKAN

```text
ten1  = GC3x3,F(tenin)
ten2  = AvgPool3x3(tenin)
ten3  = GC1x1,F(tenin)
ten4  = ten1 - ten2
ten5  = ten2 + ten3
ten6  = GELU(ten4) - GELU(ten5)
ten7  = Swish(ten4) - Swish(ten5)
ten8  = BN(DepthConcat(ten6,ten7))
ten9  = C1x1,F(GELU(GC1x1,4F(ten8)))
ten10 = C1x1,F(GELU(GC1x1,4F(tenin)))
ten11 = BN(ten9 + ten10)
tenNK = tenin - ten11
```

### Negative-Down

```text
tenND = BN(
    C3x3,s2,2F(BN(tenin))
    -
    C2x2,s2,2F(BN(tenin))
)
```

### Output

```text
CLS = Softmax(FC(GAP(tenLast), NClass))
```

## Implementation-level choices

### Group counts in grouped convolutions

The manuscript specifies grouped convolutions but does not report the numerical group count for each grouped-convolution layer.

The default implementation uses the maximum valid grouping:

```matlab
NumGroups = gcd(inputChannels,outputChannels);
```

This produces a depthwise-like grouped operator when input and output widths permit it.

### Negative-Down grouping

The manuscript equation denotes the Negative-Down branches using convolution operators but does not explicitly state the numerical grouping of those stride-2 convolutions.

The default implementation uses grouped stride-2 convolutions. Under the same maximal-grouping rule, the analytic trainable-parameter count is approximately **6.364 million**, which is consistent with the manuscript's rounded **6.4 million** two-class model size.

The behavior is configurable through:

```matlab
cfg.Model.UseGroupedDownsample
```

### Augmentation realization

The manuscript specifies augmentation types and parameter ranges but does not contain the exact random draw used for each generated image.

The repository therefore:

- fixes augmentation randomness with the configured seed;
- implements the reported operations and ranges;
- generates a deterministic sequence of 25 variants per training image.

This reproduces the stated augmentation protocol but should not be interpreted as bit-for-bit reconstruction of a historical augmented-image directory unless those original files are supplied.

### External representative-slice selection

The manuscript reports label-blinded representative sagittal slice selection for the external evaluation, but it does not fully define an automated slice-ranking algorithm.

`run_external_validation.m` therefore expects already prepared representative sagittal slices. The repository deliberately does not use ACL labels to invent a slice-selection rule.

### Expert ROI annotations

Quantitative image-level attribution metrics require the expert reference masks used in the study.

Those masks are not reconstructed from the manuscript. The XAI code accepts expert masks when they are available.

### Advanced XAI implementations

The repository provides directly executable Grad-CAM and occlusion-sensitivity utilities. The manuscript parameter settings for Score-CAM, Integrated Gradients, and LIME are preserved in the configuration so that historical or independently implemented scripts can use the same settings.

The repository does not claim that a newly reconstructed advanced-XAI implementation is identical to the historical analysis unless validated against the original outputs.

## Validation principle

The repository follows the central leakage-control rule used throughout the manuscript:

> Patient-level partitioning is completed before augmentation, and held-out patients are excluded from training, feature selection, and classifier fitting.

For DFE experiments, CWINCA is fit only on training features and the selected feature indices are frozen before the held-out set is processed.

## Release interpretation

This repository is a clean, manuscript-grounded reproducibility implementation. It is intended to make the algorithmic workflow inspectable and executable while clearly identifying details that cannot be uniquely reconstructed from the manuscript alone.

If historical source scripts are later recovered, they should be compared against this implementation before any claim of bit-for-bit equivalence is made.
