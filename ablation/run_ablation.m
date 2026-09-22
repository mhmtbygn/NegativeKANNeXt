function tbl = run_ablation(trainImds,valImds,testImds,cfg)
%RUN_ABLATION Re-train selected manuscript ablation variants.
%
% Implemented directly:
% Full, A1(all subtraction operators replaced by addition),
% A2(no negative shortcut), A3(GELU only), A4(Swish only), A5(end-to-end).
%
% A6 is obtained by running DFE on Full.
% A7 (true concat + 1x1 restoration) and A8 may require explicit graph
% rewrite if exact historical implementation is required.
%
% B0-B4 depend on "plain block" definitions that are not fully specified
% in the manuscript and are therefore intentionally not fabricated here.

variants = {
    "Full",      "subtract","dual", true;
    "A1_Add",   "add",     "dual", true;
    "A2_NoNeg", "subtract","dual", false;
    "A3_GELU",  "subtract","gelu", true;
    "A4_Swish", "subtract","swish",true
};

rows=cell(size(variants,1),5);
for i=1:size(variants,1)
    c=cfg;
    c.Model.Fusion=variants{i,2};
    c.Model.ActivationMode=variants{i,3};
    c.Model.UseNegativeShortcut=variants{i,4};

    [net,~]=train_holdout(trainImds,valImds,c);
    ev=evaluate_network(net,testImds,c);
    s=ev.metrics.scalar;
    rows(i,:)={variants{i,1},s.Accuracy,s.F1,s.MCC,c.Model.Fusion};
end

tbl=cell2table(rows,"VariableNames",["Variant","Accuracy","F1","MCC","Fusion"]);
writetable(tbl,fullfile(cfg.ResultsRoot,"ablation_core.csv"));
end
