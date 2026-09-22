%% NegativeKANNeXt complete hold-out workflow
clear; clc;
addpath(genpath(fileparts(mfilename("fullpath"))));

cfg = defaultConfig();
rng(cfg.Seed,"twister");

fprintf("1) Preparing patient-level split...\n");
split = prepare_dataset(cfg);

fprintf("2) Generating/loading training augmentation...\n");
augTrain = augment_training_set(split.train,cfg);

fprintf("3) Building and training NegativeKANNeXt...\n");
[net,trainInfo] = train_holdout(augTrain,split.val,cfg);
save(fullfile(cfg.ResultsRoot,"NegativeKANNeXt_holdout.mat"),"net","trainInfo","cfg","-v7.3");

fprintf("4) Evaluating held-out test partition...\n");
holdout = evaluate_network(net,split.test,cfg);
writetable(holdout.summary,fullfile(cfg.ResultsRoot,"holdout_metrics.csv"));

fprintf("5) Running DFE + CWINCA + shallow classifiers...\n");
dfe = run_dfe_pipeline(net,augTrain,split.test,cfg);
save(fullfile(cfg.ResultsRoot,"DFE_holdout.mat"),"dfe","-v7.3");
writetable(dfe.metrics,fullfile(cfg.ResultsRoot,"dfe_metrics.csv"));

fprintf("6) Calibration analysis...\n");
cal = calibration_analysis(holdout.scores,holdout.trueLabels,holdout.classNames,cfg);
save(fullfile(cfg.ResultsRoot,"calibration.mat"),"cal");

fprintf("Complete. Results: %s\n",cfg.ResultsRoot);
