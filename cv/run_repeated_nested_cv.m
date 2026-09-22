function results = run_repeated_nested_cv(cfg)
%RUN_REPEATED_NESTED_CV Five repetitions of stratified 5-fold patient-level CV.
%
% IMPORTANT: This is computationally expensive. Each outer fold retrains the
% deep network from scratch and applies augmentation only to outer-training patients.

rng(cfg.Seed,"twister");
imds=imageDatastore(cfg.DataRoot,"IncludeSubfolders",true,"LabelSource","foldernames");
imds.Labels=categorical(string(imds.Labels),cfg.ClassNames);

rows={};
runId=0;

for rep=1:cfg.CV.NumRepeats
    rng(cfg.Seed+rep-1,"twister");
    cvp=cvpartition(imds.Labels,"KFold",cfg.CV.NumFolds);

    for fold=1:cfg.CV.NumFolds
        runId=runId+1;
        tr=subset(imds,find(training(cvp,fold)));
        te=subset(imds,find(test(cvp,fold)));

        % Split an internal validation subset from outer training only.
        [trNet,valNet]=stratifiedInternalValidation(tr,0.10,cfg.Seed+1000*rep+fold);

        localCfg=cfg;
        localCfg.Augmentation.OutputDir=fullfile(cfg.WorkRoot, ...
            sprintf("cv_rep%02d_fold%02d_aug",rep,fold));

        aug=augment_training_set(trNet,localCfg);
        [net,~]=train_holdout(aug,valNet,localCfg);
        e2e=evaluate_network(net,te,localCfg);

        % Nested DFE: selection/classifier fitting only on outer training representation.
        dfe=run_dfe_pipeline(net,aug,te,localCfg);

        s=e2e.metrics.scalar;
        rows(end+1,:)={rep,fold,"EndToEnd",s.Accuracy,s.Sensitivity,s.Specificity,s.F1,s.GeometricMean,s.MCC}; %#ok<AGROW>

        for i=1:height(dfe.metrics)
            t=dfe.metrics(i,:);
            rows(end+1,:)={rep,fold,"DFE_"+string(t.Classifier), ...
                t.Accuracy,t.Sensitivity,t.Specificity,t.F1,t.GeometricMean,t.MCC}; %#ok<AGROW>
        end
    end
end

results=cell2table(rows,"VariableNames", ...
    ["Repeat","Fold","Model","Accuracy","Sensitivity","Specificity","F1","GeometricMean","MCC"]);
writetable(results,fullfile(cfg.ResultsRoot,"repeated_nested_cv.csv"));
end

function [tr,val]=stratifiedInternalValidation(imds,valFrac,seed)
rng(seed,"twister");
labs=string(imds.Labels);
trIdx=[]; valIdx=[];
for c=unique(labs)'
    idx=find(labs==c);
    idx=idx(randperm(numel(idx)));
    nv=max(1,round(valFrac*numel(idx)));
    valIdx=[valIdx;idx(1:nv)]; %#ok<AGROW>
    trIdx=[trIdx;idx(nv+1:end)]; %#ok<AGROW>
end
tr=subset(imds,trIdx); val=subset(imds,valIdx);
end
