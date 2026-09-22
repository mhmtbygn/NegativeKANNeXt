function T = representation_analysis(netSub,netAdd,testImds,cfg)
%REPRESENTATION_ANALYSIS Post-training branch correlation and activation statistics.
%
% Computes quantities that can be derived from saved trained networks.
% Gradient norms over the final training epoch require gradient logging
% during training and are not reconstructed post hoc.

ds=augmentedImageDatastore(cfg.InputSize(1:2),testImds, ...
    "ColorPreprocessing","gray2rgb");

rows={};
for b=1:4
    p="nk"+b;
    names=[p+"_gc3",p+"_avg",p+"_out"];

    A1=activations(netSub,ds,names(1),"OutputAs","rows");
    A2=activations(netSub,ds,names(2),"OutputAs","rows");
    O =activations(netSub,ds,names(3),"OutputAs","rows");

    B1=activations(netAdd,ds,names(1),"OutputAs","rows");
    B2=activations(netAdd,ds,names(2),"OutputAs","rows");
    BO=activations(netAdd,ds,names(3),"OutputAs","rows");

    corrSub=meanRowCorr(A1,A2);
    corrAdd=meanRowCorr(B1,B2);
    eSub=mean(O.^2,"all");
    eAdd=mean(BO.^2,"all");
    zSub=mean(abs(O)<0.01,"all");
    zAdd=mean(abs(BO)<0.01,"all");

    rows(end+1,:)={b,corrSub,corrAdd,eSub,eAdd,zSub,zAdd}; %#ok<AGROW>
end

T=cell2table(rows,"VariableNames", ...
    ["Block","AbsCorr_Sub","AbsCorr_Add","Energy_Sub","Energy_Add","NearZero_Sub","NearZero_Add"]);
writetable(T,fullfile(cfg.ResultsRoot,"representation_analysis.csv"));
end

function c=meanRowCorr(A,B)
n=size(A,1); vals=zeros(n,1);
for i=1:n
    a=double(A(i,:)); b=double(B(i,:));
    R=corrcoef(a,b);
    if all(size(R)==[2 2]) && isfinite(R(1,2))
        vals(i)=abs(R(1,2));
    else
        vals(i)=NaN;
    end
end
c=mean(vals,"omitnan");
end
