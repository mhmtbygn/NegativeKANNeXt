function out = run_dfe_pipeline(net,trainImds,testImds,cfg)
%RUN_DFE_PIPELINE GAP -> CWINCA -> six shallow classifiers.

trainDS=augmentedImageDatastore(cfg.InputSize(1:2),trainImds, ...
    "ColorPreprocessing","gray2rgb");
testDS=augmentedImageDatastore(cfg.InputSize(1:2),testImds, ...
    "ColorPreprocessing","gray2rgb");

Xtr=activations(net,trainDS,"gap","OutputAs","rows", ...
    "MiniBatchSize",cfg.Training.MiniBatchSize);
Xte=activations(net,testDS,"gap","OutputAs","rows", ...
    "MiniBatchSize",cfg.Training.MiniBatchSize);

if size(Xtr,2)~=768
    warning("Expected 768 GAP features; got %d.",size(Xtr,2));
end

if cfg.DFE.UseReportedIndices
    sel.indices=cfg.DFE.FixedReportedIndices;
    sel.xmin=min(double(Xtr),[],1);
    sel.xmax=max(double(Xtr),[],1);
    sel.epsilon=1e-12;
    sel.bestK=numel(sel.indices);
    sel.bestLoss=NaN;
else
    sel=cwinca_select(Xtr,trainImds.Labels,cfg);
end

XtrS=apply_cwinca(Xtr,sel);
XteS=apply_cwinca(Xte,sel);

models=train_shallow_classifiers(XtrS,trainImds.Labels);

names=fieldnames(models);
rows=cell(numel(names),10);
predictions=struct();

for i=1:numel(names)
    name=names{i};
    mdl=models.(name);
    [pred,score]=predict(mdl,XteS);
    cls=string(mdl.ClassNames);

    met=classification_metrics(testImds.Labels,pred,score,cls,cfg.PositiveClass);
    s=met.scalar;
    rows(i,:)={name,s.Accuracy,s.Sensitivity,s.Specificity,s.Precision, ...
        s.F1,s.AUC,s.GeometricMean,s.MCC,sel.bestK};
    predictions.(name)=struct("labels",pred,"scores",score,"classes",cls);
end

out.metrics=cell2table(rows,"VariableNames", ...
    ["Classifier","Accuracy","Sensitivity","Specificity","Precision", ...
     "F1","AUC","GeometricMean","MCC","NumFeatures"]);
out.selector=sel;
out.models=models;
out.predictions=predictions;
out.trainFeatures=Xtr;
out.testFeatures=Xte;
end
