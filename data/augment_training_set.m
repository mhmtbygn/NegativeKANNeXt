function augImds = augment_training_set(trainImds,cfg)
%AUGMENT_TRAINING_SET Generate exactly 25 deterministic variants per training image.
%
% The original image itself is not added because manuscript counts
% (222*25=5550 and 255*25=6375) correspond to 25 generated variants.

outRoot = cfg.Augmentation.OutputDir;
manifestFile = fullfile(outRoot,"augmentation_manifest.csv");

if exist(manifestFile,"file")
    augImds = imageDatastore(outRoot,"IncludeSubfolders",true,"LabelSource","foldernames");
    augImds.Labels = categorical(string(augImds.Labels),cfg.ClassNames);
    return;
end

if exist(outRoot,"dir"), rmdir(outRoot,"s"); end
mkdir(outRoot);

rng(cfg.Augmentation.Seed,"twister");

rows = {};
for i = 1:numel(trainImds.Files)
    I = readimage(trainImds,i);
    I = prepareRGB(I,cfg.InputSize(1:2));
    lab = string(trainImds.Labels(i));
    labDir = fullfile(outRoot,lab);
    if ~exist(labDir,"dir"), mkdir(labDir); end

    [~,base,~] = fileparts(trainImds.Files{i});

    for v = 1:cfg.Augmentation.NumVariants
        J = applyVariant(I,v);
        fn = sprintf("%s_aug%02d.png",base,v);
        fp = fullfile(labDir,fn);
        imwrite(J,fp);
        rows(end+1,:) = {trainImds.Files{i},fp,lab,v}; %#ok<AGROW>
    end
end

T = cell2table(rows,"VariableNames",["SourceFile","AugmentedFile","Label","Variant"]);
writetable(T,manifestFile);

augImds = imageDatastore(outRoot,"IncludeSubfolders",true,"LabelSource","foldernames");
augImds.Labels = categorical(string(augImds.Labels),cfg.ClassNames);
end

function I = prepareRGB(I,targetHW)
if size(I,3)==1, I = repmat(I,1,1,3); end
I = imresize(I,targetHW);
if ~isa(I,"uint8")
    I = im2uint8(mat2gray(I));
end
end

function J = applyVariant(I,v)
% Deterministic recipe family; random values are controlled by cfg seed.
switch v
    case 1,  J = rot(I, randRange(40,75));
    case 2,  J = rot(I,-randRange(40,75));
    case 3,  J = rot(I, randRange(100,130));
    case 4,  J = rot(I,-randRange(100,130));
    case 5,  J = fliplr(I);
    case 6,  J = flipud(I);
    case 7,  J = gammaAdj(I,2.0);
    case 8,  J = gammaAdj(I,0.4);
    case 9,  J = imnoise(I,"gaussian",0,0.05);
    case 10, J = imnoise(I,"salt & pepper",0.05);
    case 11, J = imgaussfilt(I,2,"FilterSize",7);
    case 12, J = motionBlur(I,15,randRange(0,180));
    case 13, J = contrastAdj(I);
    case 14, J = channelPerturb(I);
    case 15, J = I(:,:,randperm(3));
    case 16, J = shearImg(I,0.18,0);
    case 17, J = shearImg(I,-0.18,0);
    case 18, J = shearImg(I,0,0.18);
    case 19, J = shearImg(I,0,-0.18);
    case 20, J = histEq3(I);
    case 21, J = imcomplement(I);
    case 22, J = imgaussfilt(rot(I,randRange(40,75)),2,"FilterSize",7);
    case 23, J = imnoise(fliplr(rot(I,-randRange(40,75))),"gaussian",0,0.05);
    case 24, J = contrastAdj(shearImg(flipud(I),0.12,0));
    case 25, J = gammaAdj(motionBlur(rot(I,randRange(100,130)),15,randRange(0,180)),0.4);
    otherwise, error("Variant index out of range.");
end
J = im2uint8(mat2gray(J));
end

function x = randRange(a,b), x = a + (b-a)*rand(); end
function J = rot(I,a), J = imrotate(I,a,"bilinear","crop"); end
function J = gammaAdj(I,g)
J = zeros(size(I),"uint8");
for k=1:3, J(:,:,k)=imadjust(I(:,:,k),[],[],g); end
end
function J = motionBlur(I,len,ang)
h = fspecial("motion",len,ang);
J = imfilter(I,h,"replicate");
end
function J = contrastAdj(I)
J = zeros(size(I),"uint8");
for k=1:3, J(:,:,k)=imadjust(I(:,:,k),stretchlim(I(:,:,k),[0.01 0.99]),[]); end
end
function J = channelPerturb(I)
X = im2double(I);
scale = 0.85 + 0.30*rand(1,1,3);
bias = -0.05 + 0.10*rand(1,1,3);
X = min(max(X.*scale + bias,0),1);
J = im2uint8(X);
end
function J = shearImg(I,sx,sy)
tform = affine2d([1 sy 0; sx 1 0; 0 0 1]);
ref = imref2d(size(I(:,:,1)));
J = imwarp(I,tform,"OutputView",ref,"FillValues",0);
end
function J = histEq3(I)
J = zeros(size(I),"uint8");
for k=1:3, J(:,:,k)=histeq(I(:,:,k)); end
end
