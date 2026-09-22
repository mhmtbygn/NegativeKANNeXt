function out = run_xai(net,testImds,cfg,expertMaskRoot)
%RUN_XAI Generate Grad-CAM and manual occlusion maps; optional expert-mask scoring.
%
% Advanced Score-CAM / Integrated Gradients / LIME settings are recorded in
% cfg. This function provides stable MATLAB-native Grad-CAM plus a manual
% occlusion implementation. Extend the same output structure for the other
% methods if exact historical scripts become available.

if nargin<4, expertMaskRoot=""; end

n=numel(testImds.Files);
out=struct();
out.items=cell(n,1);

featureLayer = findLastConvLayer(net);

for i=1:n
    I=readimage(testImds,i);
    I=prepareInput(I,cfg.InputSize(1:2));

    [lab,score]=classify(net,I);
    gcam=gradCAM(net,I,lab,"FeatureLayer",featureLayer);
    occ=manualOcclusion(net,I,lab,cfg.XAI.OcclusionMaskComparison,cfg.XAI.OcclusionStrideComparison);

    item=struct("file",testImds.Files{i},"label",lab,"score",score, ...
        "gradCAM",gcam,"occlusion",occ);

    if strlength(expertMaskRoot)>0
        [~,b,~]=fileparts(testImds.Files{i});
        mp=fullfile(expertMaskRoot,b+".png");
        if exist(mp,"file")
            M=imbinarize(imresize(imread(mp),cfg.InputSize(1:2)));
            item.gradCAM_IoU=iouThreshold(gcam,M,0.70);
            item.occlusion_IoU=iouThreshold(occ,M,0.70);
        end
    end
    out.items{i}=item;
end
end

function name=findLastConvLayer(net)
name="";
for k=numel(net.Layers):-1:1
    if isa(net.Layers(k),"nnet.cnn.layer.Convolution2DLayer") || ...
       isa(net.Layers(k),"nnet.cnn.layer.GroupedConvolution2DLayer")
        name=net.Layers(k).Name; return;
    end
end
error("No convolutional layer found.");
end

function I=prepareInput(I,hw)
if size(I,3)==1, I=repmat(I,1,1,3); end
I=imresize(I,hw);
end

function map=manualOcclusion(net,I,targetLabel,maskSize,stride)
base=classScore(net,I,targetLabel);
H=size(I,1); W=size(I,2);
acc=zeros(H,W); cnt=zeros(H,W);
mh=maskSize(1); mw=maskSize(2); sh=stride(1); sw=stride(2);
for r=1:sh:max(1,H-mh+1)
    for c=1:sw:max(1,W-mw+1)
        J=I;
        J(r:min(H,r+mh-1),c:min(W,c+mw-1),:)=0;
        s=classScore(net,J,targetLabel);
        d=max(0,base-s);
        rr=r:min(H,r+mh-1); cc=c:min(W,c+mw-1);
        acc(rr,cc)=acc(rr,cc)+d;
        cnt(rr,cc)=cnt(rr,cc)+1;
    end
end
map=acc./max(cnt,1);
map=rescale(map);
end

function s=classScore(net,I,label)
[~,scores]=classify(net,I);
classes=string(net.Layers(end).Classes);
idx=find(classes==string(label),1);
s=scores(idx);
end

function v=iouThreshold(A,M,pct)
t=prctile(A(:),100*pct);
B=A>=t;
v=sum(B(:)&M(:))/max(sum(B(:)|M(:)),1);
end
