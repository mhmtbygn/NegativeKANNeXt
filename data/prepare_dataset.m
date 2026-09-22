function split = prepare_dataset(cfg)
%PREPARE_DATASET Exact stratified patient-level split before augmentation.

rng(cfg.Seed,"twister");

imds = imageDatastore(cfg.DataRoot, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

% Normalize labels to strings for validation.
labs = string(imds.Labels);
expected = cfg.ClassNames;
for c = expected
    if ~any(labs == c)
        error("Missing expected class folder/label: %s",c);
    end
end

idxACL = find(labs == "ACL_Rupture");
idxCTL = find(labs == "Control");

if numel(idxACL) ~= 317 || numel(idxCTL) ~= 364
    warning("Manuscript cohort expected 317 ACL and 364 Control patients; found %d and %d.", ...
        numel(idxACL),numel(idxCTL));
end

idxACL = idxACL(randperm(numel(idxACL)));
idxCTL = idxCTL(randperm(numel(idxCTL)));

a = cfg.Split.ACL;
c = cfg.Split.Control;

split.train = subset(imds,[ ...
    idxACL(1:a.Train); ...
    idxCTL(1:c.Train)]);

split.val = subset(imds,[ ...
    idxACL(a.Train+(1:a.Validation)); ...
    idxCTL(c.Train+(1:c.Validation))]);

split.test = subset(imds,[ ...
    idxACL(a.Train+a.Validation+(1:a.Test)); ...
    idxCTL(c.Train+c.Validation+(1:c.Test))]);

split.train.Labels = categorical(string(split.train.Labels),cfg.ClassNames);
split.val.Labels   = categorical(string(split.val.Labels),cfg.ClassNames);
split.test.Labels  = categorical(string(split.test.Labels),cfg.ClassNames);

split.seed = cfg.Seed;
split.counts = table( ...
    [sum(split.train.Labels=="ACL_Rupture");sum(split.train.Labels=="Control")], ...
    [sum(split.val.Labels=="ACL_Rupture");sum(split.val.Labels=="Control")], ...
    [sum(split.test.Labels=="ACL_Rupture");sum(split.test.Labels=="Control")], ...
    "VariableNames",["Train","Validation","Test"], ...
    "RowNames",["ACL_Rupture","Control"]);

disp(split.counts);
end
