function cfg = defaultConfig()
%DEFAULTCONFIG Central configuration for NegativeKANNeXt experiments.

cfg.Seed = 42;
cfg.DataRoot = fullfile(pwd,"data_original");
cfg.WorkRoot = fullfile(pwd,"work");
cfg.ResultsRoot = fullfile(pwd,"results");

cfg.ClassNames = ["ACL_Rupture","Control"];
cfg.PositiveClass = "ACL_Rupture";
cfg.InputSize = [224 224 3];

% Exact manuscript patient counts.
cfg.Split.ACL = struct("Train",222,"Validation",32,"Test",63);
cfg.Split.Control = struct("Train",255,"Validation",36,"Test",73);

cfg.Augmentation.NumVariants = 25;
cfg.Augmentation.Seed = 42;
cfg.Augmentation.OutputDir = fullfile(cfg.WorkRoot,"augmented_train");

% Architecture.
cfg.Model.Widths = [96 192 384 768];
cfg.Model.NumClasses = 2;
cfg.Model.MaximalGrouping = true;
cfg.Model.UseGroupedDownsample = true; % parameter-count-consistent interpretation
cfg.Model.Fusion = "subtract";         % subtract | add | concat | residual
cfg.Model.ActivationMode = "dual";     % dual | gelu | swish
cfg.Model.UseNegativeShortcut = true;

% Training.
cfg.Training.InitialLearnRate = 0.01;
cfg.Training.Momentum = 0.9;
cfg.Training.L2Regularization = 1e-4;
cfg.Training.MiniBatchSize = 200;
cfg.Training.MaxEpochs = 40;
cfg.Training.ValidationFrequency = 50;
cfg.Training.ExecutionEnvironment = "auto";
cfg.Training.Verbose = true;
cfg.Training.Plots = "training-progress";

% DFE / CWINCA.
cfg.DFE.LowerThreshold = 0.05;
cfg.DFE.UpperThreshold = 0.99;
cfg.DFE.InnerFolds = 10;
cfg.DFE.FixedReportedIndices = [48 571 141 338];
cfg.DFE.UseReportedIndices = false;

% Repeated patient-level CV.
cfg.CV.NumFolds = 5;
cfg.CV.NumRepeats = 5;

% Calibration.
cfg.Calibration.Edges = [0.5 0.6 0.7 0.8 0.9 1.0];

% XAI.
cfg.XAI.OcclusionMaskRepresentative = [24 24];
cfg.XAI.OcclusionStrideRepresentative = [12 12];
cfg.XAI.OcclusionMaskDetailed = [16 16];
cfg.XAI.OcclusionStrideDetailed = [8 8];
cfg.XAI.OcclusionMaskComparison = [24 24];
cfg.XAI.OcclusionStrideComparison = [8 8];
cfg.XAI.IntegratedGradientsSteps = 50;
cfg.XAI.ScoreCAMBatchSize = 64;
cfg.XAI.ScoreCAMRangeThreshold = 1e-8;
cfg.XAI.LIMENumFeatures = 64;
cfg.XAI.LIMENumSamples = 3072;

if ~exist(cfg.WorkRoot,"dir"), mkdir(cfg.WorkRoot); end
if ~exist(cfg.ResultsRoot,"dir"), mkdir(cfg.ResultsRoot); end
end
