function lgraph = buildNegativeKANNeXt(cfg)
%BUILDNEGATIVEKANNEXT Build the manuscript-defined NegativeKANNeXt graph.

W = cfg.Model.Widths;
inSize = cfg.InputSize;

lgraph = layerGraph();
input = imageInputLayer(inSize,"Name","input","Normalization","zerocenter");
lgraph = addLayers(lgraph,input);

% ---------------- Negative-Stem ----------------
stem1 = [
    convolution2dLayer(7,W(1),"Stride",4,"Padding","same", ...
        "Name","stem_conv7","WeightsInitializer","glorot","BiasInitializer","zeros")
    functionLayer(@geluFn,"Name","stem_gelu","Formattable",true)
];
stem2 = convolution2dLayer(4,W(1),"Stride",4,"Padding","same", ...
    "Name","stem_conv4","WeightsInitializer","glorot","BiasInitializer","zeros");
stemSub = binaryFusionLayer("stem_fusion","subtract");
stemBN = batchNormalizationLayer("Name","stem_bn");

lgraph = addLayers(lgraph,stem1);
lgraph = addLayers(lgraph,stem2);
lgraph = addLayers(lgraph,stemSub);
lgraph = addLayers(lgraph,stemBN);

lgraph = connectLayers(lgraph,"input","stem_conv7");
lgraph = connectLayers(lgraph,"input","stem_conv4");
lgraph = connectLayers(lgraph,"stem_gelu","stem_fusion/in1");
lgraph = connectLayers(lgraph,"stem_conv4","stem_fusion/in2");
lgraph = connectLayers(lgraph,"stem_fusion","stem_bn");

current = "stem_bn";

for s = 1:4
    F = W(s);
    [lgraph,current] = addNegativeKAN(lgraph,current,F,"nk"+s,cfg);

    if s < 4
        [lgraph,current] = addNegativeDown(lgraph,current,F,W(s+1),"nd"+s,cfg);
    end
end

head = [
    globalAveragePooling2dLayer("Name","gap")
    fullyConnectedLayer(cfg.Model.NumClasses,"Name","fc", ...
        "WeightsInitializer","glorot","BiasInitializer","zeros")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")
];
lgraph = addLayers(lgraph,head);
lgraph = connectLayers(lgraph,current,"gap");
end

function [g,out] = addNegativeKAN(g,in,F,prefix,cfg)
% Branch 1: grouped 3x3
b1 = groupedConv(3,F,F,1,"same",prefix+"_gc3");
% Branch 2: 3x3 average pooling
b2 = averagePooling2dLayer(3,"Stride",1,"Padding","same","Name",prefix+"_avg");
% Branch 3: grouped 1x1
b3 = groupedConv(1,F,F,1,"same",prefix+"_gc1");

g=addLayers(g,b1); g=addLayers(g,b2); g=addLayers(g,b3);
g=connectLayers(g,in,prefix+"_gc3");
g=connectLayers(g,in,prefix+"_avg");
g=connectLayers(g,in,prefix+"_gc1");

f14 = binaryFusionLayer(prefix+"_ten4",cfg.Model.Fusion);
add5 = binaryFusionLayer(prefix+"_ten5","add");
g=addLayers(g,f14); g=addLayers(g,add5);
g=connectLayers(g,prefix+"_gc3",prefix+"_ten4/in1");
g=connectLayers(g,prefix+"_avg",prefix+"_ten4/in2");
g=connectLayers(g,prefix+"_avg",prefix+"_ten5/in1");
g=connectLayers(g,prefix+"_gc1",prefix+"_ten5/in2");

% Dual/single activation difference paths.
if cfg.Model.ActivationMode == "dual"
    a6a=functionLayer(@geluFn,"Name",prefix+"_gelu4","Formattable",true);
    a6b=functionLayer(@geluFn,"Name",prefix+"_gelu5","Formattable",true);
    a7a=functionLayer(@swishFn,"Name",prefix+"_swish4","Formattable",true);
    a7b=functionLayer(@swishFn,"Name",prefix+"_swish5","Formattable",true);
    d6=binaryFusionLayer(prefix+"_ten6",cfg.Model.Fusion);
    d7=binaryFusionLayer(prefix+"_ten7",cfg.Model.Fusion);
    cat=depthConcatenationLayer(2,"Name",prefix+"_cat");
    g=addLayers(g,a6a);g=addLayers(g,a6b);g=addLayers(g,a7a);g=addLayers(g,a7b);
    g=addLayers(g,d6);g=addLayers(g,d7);g=addLayers(g,cat);
    g=connectLayers(g,prefix+"_ten4",prefix+"_gelu4");
    g=connectLayers(g,prefix+"_ten5",prefix+"_gelu5");
    g=connectLayers(g,prefix+"_ten4",prefix+"_swish4");
    g=connectLayers(g,prefix+"_ten5",prefix+"_swish5");
    g=connectLayers(g,prefix+"_gelu4",prefix+"_ten6/in1");
    g=connectLayers(g,prefix+"_gelu5",prefix+"_ten6/in2");
    g=connectLayers(g,prefix+"_swish4",prefix+"_ten7/in1");
    g=connectLayers(g,prefix+"_swish5",prefix+"_ten7/in2");
    g=connectLayers(g,prefix+"_ten6",prefix+"_cat/in1");
    g=connectLayers(g,prefix+"_ten7",prefix+"_cat/in2");
    catChannels = 2*F;
    catOut = prefix+"_cat";
elseif cfg.Model.ActivationMode == "gelu"
    a1=functionLayer(@geluFn,"Name",prefix+"_gelu4","Formattable",true);
    a2=functionLayer(@geluFn,"Name",prefix+"_gelu5","Formattable",true);
    d=binaryFusionLayer(prefix+"_ten6",cfg.Model.Fusion);
    g=addLayers(g,a1);g=addLayers(g,a2);g=addLayers(g,d);
    g=connectLayers(g,prefix+"_ten4",prefix+"_gelu4");
    g=connectLayers(g,prefix+"_ten5",prefix+"_gelu5");
    g=connectLayers(g,prefix+"_gelu4",prefix+"_ten6/in1");
    g=connectLayers(g,prefix+"_gelu5",prefix+"_ten6/in2");
    catChannels=F; catOut=prefix+"_ten6";
elseif cfg.Model.ActivationMode == "swish"
    a1=functionLayer(@swishFn,"Name",prefix+"_swish4","Formattable",true);
    a2=functionLayer(@swishFn,"Name",prefix+"_swish5","Formattable",true);
    d=binaryFusionLayer(prefix+"_ten7",cfg.Model.Fusion);
    g=addLayers(g,a1);g=addLayers(g,a2);g=addLayers(g,d);
    g=connectLayers(g,prefix+"_ten4",prefix+"_swish4");
    g=connectLayers(g,prefix+"_ten5",prefix+"_swish5");
    g=connectLayers(g,prefix+"_swish4",prefix+"_ten7/in1");
    g=connectLayers(g,prefix+"_swish5",prefix+"_ten7/in2");
    catChannels=F; catOut=prefix+"_ten7";
else
    error("Unknown ActivationMode.");
end

bn8=batchNormalizationLayer("Name",prefix+"_bn8");
g=addLayers(g,bn8); g=connectLayers(g,catOut,prefix+"_bn8");

% ten9 = C1x1,F( GELU( GC1x1,4F(ten8) ) )
e9=groupedConv(1,catChannels,4*F,1,"same",prefix+"_exp9");
a9=functionLayer(@geluFn,"Name",prefix+"_gelu9","Formattable",true);
c9=convolution2dLayer(1,F,"Padding","same","Name",prefix+"_proj9", ...
    "WeightsInitializer","glorot","BiasInitializer","zeros");
g=addLayers(g,e9);g=addLayers(g,a9);g=addLayers(g,c9);
g=connectLayers(g,prefix+"_bn8",prefix+"_exp9");
g=connectLayers(g,prefix+"_exp9",prefix+"_gelu9");
g=connectLayers(g,prefix+"_gelu9",prefix+"_proj9");

% ten10 = C1x1,F( GELU( GC1x1,4F(tenin) ) )
e10=groupedConv(1,F,4*F,1,"same",prefix+"_exp10");
a10=functionLayer(@geluFn,"Name",prefix+"_gelu10","Formattable",true);
c10=convolution2dLayer(1,F,"Padding","same","Name",prefix+"_proj10", ...
    "WeightsInitializer","glorot","BiasInitializer","zeros");
g=addLayers(g,e10);g=addLayers(g,a10);g=addLayers(g,c10);
g=connectLayers(g,in,prefix+"_exp10");
g=connectLayers(g,prefix+"_exp10",prefix+"_gelu10");
g=connectLayers(g,prefix+"_gelu10",prefix+"_proj10");

a11=binaryFusionLayer(prefix+"_add11","add");
bn11=batchNormalizationLayer("Name",prefix+"_bn11");
g=addLayers(g,a11);g=addLayers(g,bn11);
g=connectLayers(g,prefix+"_proj9",prefix+"_add11/in1");
g=connectLayers(g,prefix+"_proj10",prefix+"_add11/in2");
g=connectLayers(g,prefix+"_add11",prefix+"_bn11");

if cfg.Model.UseNegativeShortcut
    outLayer=binaryFusionLayer(prefix+"_out",cfg.Model.Fusion);
    g=addLayers(g,outLayer);
    g=connectLayers(g,in,prefix+"_out/in1");
    g=connectLayers(g,prefix+"_bn11",prefix+"_out/in2");
    out=prefix+"_out";
else
    out=prefix+"_bn11";
end
end

function [g,out] = addNegativeDown(g,in,Fin,Fout,prefix,cfg)
pre=batchNormalizationLayer("Name",prefix+"_prebn");
g=addLayers(g,pre);g=connectLayers(g,in,prefix+"_prebn");

if cfg.Model.UseGroupedDownsample
    c3=groupedConv(3,Fin,Fout,2,"same",prefix+"_conv3");
    c2=groupedConv(2,Fin,Fout,2,"same",prefix+"_conv2");
else
    c3=convolution2dLayer(3,Fout,"Stride",2,"Padding","same", ...
        "Name",prefix+"_conv3","WeightsInitializer","glorot","BiasInitializer","zeros");
    c2=convolution2dLayer(2,Fout,"Stride",2,"Padding","same", ...
        "Name",prefix+"_conv2","WeightsInitializer","glorot","BiasInitializer","zeros");
end
f=binaryFusionLayer(prefix+"_fusion",cfg.Model.Fusion);
post=batchNormalizationLayer("Name",prefix+"_postbn");
g=addLayers(g,c3);g=addLayers(g,c2);g=addLayers(g,f);g=addLayers(g,post);
g=connectLayers(g,prefix+"_prebn",prefix+"_conv3");
g=connectLayers(g,prefix+"_prebn",prefix+"_conv2");
g=connectLayers(g,prefix+"_conv3",prefix+"_fusion/in1");
g=connectLayers(g,prefix+"_conv2",prefix+"_fusion/in2");
g=connectLayers(g,prefix+"_fusion",prefix+"_postbn");
out=prefix+"_postbn";
end

function L = groupedConv(k,inC,outC,stride,padding,name)
ng = gcd(inC,outC);
filtersPerGroup = outC/ng;
L = groupedConvolution2dLayer(k,filtersPerGroup,ng, ...
    "Stride",stride,"Padding",padding,"Name",name, ...
    "WeightsInitializer","glorot","BiasInitializer","zeros");
end

function L = binaryFusionLayer(name,mode)
switch string(mode)
    case "subtract"
        L = functionLayer(@(a,b)a-b,"Name",name,"NumInputs",2,"Formattable",true);
    case "add"
        L = functionLayer(@(a,b)a+b,"Name",name,"NumInputs",2,"Formattable",true);
    case "residual"
        L = functionLayer(@(a,b)a+b,"Name",name,"NumInputs",2,"Formattable",true);
    case "concat"
        % For generic fusion replacement, addition is not dimension-changing.
        % True A7 is built in run_ablation with a graph rewrite.
        L = functionLayer(@(a,b)a-b,"Name",name,"NumInputs",2,"Formattable",true);
    otherwise
        error("Unsupported fusion mode: %s",mode);
end
end

function y = geluFn(x)
y = 0.5*x.*(1+erf(x/sqrt(2)));
end

function y = swishFn(x)
y = x./(1+exp(-x));
end
