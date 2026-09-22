function out = evaluate_network(net,testImds,cfg)
%EVALUATE_NETWORK Held-out, non-augmented patient-level evaluation.

testDS = augmentedImageDatastore(cfg.InputSize(1:2),testImds, ...
    "ColorPreprocessing","gray2rgb");

[pred,scores] = classify(net,testDS, ...
    "MiniBatchSize",cfg.Training.MiniBatchSize, ...
    "ExecutionEnvironment",cfg.Training.ExecutionEnvironment);

trueLab = testImds.Labels;
classNames = string(net.Layers(end).Classes);

M = classification_metrics(trueLab,pred,scores,classNames,cfg.PositiveClass);

out.summary = struct2table(M.scalar);
out.metrics = M;
out.predictedLabels = pred;
out.trueLabels = trueLab;
out.scores = scores;
out.classNames = classNames;
end
