# Assumptions and reproducibility scope

This file separates details explicitly supported by the manuscript from implementation choices required to turn the manuscript description into executable code.

## Directly specified in the manuscript

- MATLAB R2023b.
- Input size: 224 x 224 x 3.
- Seed for the principal 70:10:20 patient-level split: 42.
- Cohort: 681 patients, one representative sagittal slice per patient.
- Training/validation/test counts: 477 / 68 / 136.
- Training augmentation: 25 variants per training image, 11,925 total training images.
- Optimizer: SGDM.
- Initial learning rate: 0.01.
- Momentum: 0.9.
- L2 regularization: 1e-4.
- Mini-batch size: 200.
- Epochs: 40.
- Constant learning rate.
- Shuffle every epoch.
- Validation frequency: 50 iterations.
- No early stopping.
- Glorot weight initialization, zero bias initialization.
- Zero-center input normalization.
- No class weighting.
- Architecture stage widths: 96, 192, 384, 768.
- GAP feature length: 768.
- CWINCA thresholds: 0.05 and 0.99.
- CWINCA loss model: 1-NN, Cityblock/L1, equal weighting, standardized, 10-fold CV.
- Final reported hold-out selected features: 48, 571, 141, 338.
- DFE classifiers: kNN, cubic SVM, decision tree, linear discriminant, kernel Naive Bayes, logistic regression.
- Calibration bins: [0.5,0.6), [0.6,0.7), [0.7,0.8), [0.8,0.9), [0.9,1.0].
- Occlusion settings used in the manuscript: 16x16/stride 8 for the detailed class-specific visualization and 24x24/stride 8 for the multi-method comparison.
- Integrated Gradients: zero baseline, 50 steps.
- LIME: 64 interpretable features, 3072 perturbations.
- Score-CAM: same final spatial layer as Grad-CAM, positive activations, range threshold 1e-8, batch size 64.

## Mathematical architecture recovered from the DOCX equations

Negative-Stem:

`tenNS = BN( GELU(C7x7,s4,96(Img)) - C4x4,s4,96(Img) )`

NegativeKAN:

- `ten1 = GC3x3,F(tenin)`
- `ten2 = AvgPool3x3(tenin)`
- `ten3 = GC1x1,F(tenin)`
- `ten4 = ten1 - ten2`
- `ten5 = ten2 + ten3`
- `ten6 = GELU(ten4) - GELU(ten5)`
- `ten7 = Swish(ten4) - Swish(ten5)`
- `ten8 = BN(DepthConcat(ten6,ten7))`
- `ten9 = C1x1,F(GELU(GC1x1,4F(ten8)))`
- `ten10 = C1x1,F(GELU(GC1x1,4F(tenin)))`
- `ten11 = BN(ten9 + ten10)`
- `tenNK = tenin - ten11`

Negative-Down:

`tenND = BN( C3x3,s2,2F(BN(tenin)) - C2x2,s2,2F(BN(tenin)) )`

Output:

`CLS = Softmax(FC(GAP(tenLast), NClass))`

## Details not uniquely determined by the manuscript

### 1. Group counts in grouped convolutions

The paper states grouped convolutions but does not provide numerical group counts. The implementation therefore uses maximum valid grouping:

`NumGroups = gcd(inputChannels, outputChannels)`.

This makes the grouped operator depthwise-like when input and output widths permit it.

### 2. Negative-Down convolution grouping

The equation uses `C(.)`, while the model is described as compact and the reported parameter count is ~6.4 M. Standard dense downsampling convolutions make the parameter count substantially larger when combined with the documented inverted bottlenecks. Therefore, the default code uses grouped stride-2 convolutions in Negative-Down. Under maximal grouping, the analytic trainable-parameter count is approximately 6.364 M, which rounds to the 6.4 M reported in the manuscript. This numerical agreement strongly supports this implementation choice. The choice remains configurable.

### 3. Exact random augmentation draws

The manuscript gives augmentation types and parameter ranges, but not the exact random draw used for each patient/variant. This package fixes augmentation randomness through `cfg.Seed` and provides a deterministic 25-recipe sequence.

### 4. Exact label-blinded MRNet representative-slice selection rule

The manuscript describes label-blinded representative slice selection for external validation but does not fully define an automated selection algorithm. `run_external_validation.m` therefore expects already prepared representative sagittal slices or a user-supplied selection function.

### 5. Exact expert ROI annotations

Quantitative XAI metrics require the expert masks used in the study. They are not contained in the manuscript. The code supports these masks when supplied, but cannot recreate them from the paper.

## Recommendation before public release

If the historical source code can be recovered, compare:

- group counts,
- Negative-Down implementation,
- augmentation recipe order,
- external representative-slice selection,

against this clean implementation. Update this file and the code if differences are found. Do not silently claim bit-for-bit reproduction until that comparison is completed.
