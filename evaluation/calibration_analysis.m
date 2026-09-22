function cal = calibration_analysis(scores,yTrue,classNames,cfg)
%CALIBRATION_ANALYSIS Confidence-based calibration from manuscript.
%
% Confidence = maximum predicted class probability.
% Correctness = binary response for slope/intercept and confidence Brier score.

[conf,idx]=max(scores,[],2);
classes=string(classNames);
pred=classes(idx);
truth=string(yTrue);
correct=double(pred(:)==truth(:));

edges=cfg.Calibration.Edges;
nb=numel(edges)-1;
count=zeros(nb,1); meanConf=nan(nb,1); obsAcc=nan(nb,1); gap=nan(nb,1);

for b=1:nb
    if b<nb
        m=conf>=edges(b) & conf<edges(b+1);
    else
        m=conf>=edges(b) & conf<=edges(b+1);
    end
    count(b)=sum(m);
    if count(b)>0
        meanConf(b)=mean(conf(m));
        obsAcc(b)=mean(correct(m));
        gap(b)=abs(meanConf(b)-obsAcc(b));
    end
end

ece=nansum((count/sum(count)).*gap);
mce=max(gap,[],"omitnan");
brier=mean((conf-correct).^2);

p=min(max(conf,1e-6),1-1e-6);
logit=log(p./(1-p));
b=glmfit(logit,correct,"binomial","link","logit");
intercept=b(1); slope=b(2);

cal = struct();
cal.BrierScore=brier;
cal.ECE=ece;
cal.MCE=mce;
cal.Slope=slope;
cal.Intercept=intercept;
cal.binTable=table(edges(1:end-1)',edges(2:end)',count,meanConf,obsAcc,gap, ...
    "VariableNames",["Lower","Upper","N","MeanConfidence","ObservedAccuracy","AbsGap"]);
end
