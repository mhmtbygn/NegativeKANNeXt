function out = run_external_validation(net,externalRoot,cfg)
%RUN_EXTERNAL_VALIDATION Frozen-model external evaluation.
%
% externalRoot must contain prepared representative sagittal slices:
% externalRoot/ACL_Rupture/*
% externalRoot/Control/*
%
% The manuscript's exact automated label-blinded representative-slice
% selector is not fully specified, so this function intentionally does not
% invent one. Prepare slices independently of ACL labels before calling.

ext=imageDatastore(externalRoot,"IncludeSubfolders",true,"LabelSource","foldernames");
ext.Labels=categorical(string(ext.Labels),cfg.ClassNames);

out=evaluate_network(net,ext,cfg);
writetable(out.summary,fullfile(cfg.ResultsRoot,"external_frozen_metrics.csv"));
end
