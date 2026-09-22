function M = classification_metrics(yTrue,yPred,scores,classNames,positiveClass)
%CLASSIFICATION_METRICS Accuracy, sensitivity, specificity, precision, F1,
%AUC, geometric mean, MCC and Wilson CI.

yt = string(yTrue);
yp = string(yPred);
pos = string(positiveClass);

tp=sum(yt==pos & yp==pos);
tn=sum(yt~=pos & yp~=pos);
fp=sum(yt~=pos & yp==pos);
fn=sum(yt==pos & yp~=pos);
n=tp+tn+fp+fn;

acc=(tp+tn)/n;
sens=tp/max(tp+fn,1);
spec=tn/max(tn+fp,1);
prec=tp/max(tp+fp,1);
f1=2*prec*sens/max(prec+sens,eps);
gm=sqrt(sens*spec);
den=sqrt(double((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)));
mcc=((tp*tn)-(fp*fn))/max(den,eps);

auc=NaN;
if ~isempty(scores)
    posCol=find(string(classNames)==pos,1);
    if ~isempty(posCol)
        [~,~,~,auc]=perfcurve(yt,scores(:,posCol),pos);
    end
end

ci=wilson_ci(tp+tn,n,0.95);

M.scalar = struct( ...
    "Accuracy",100*acc, ...
    "Sensitivity",100*sens, ...
    "Specificity",100*spec, ...
    "Precision",100*prec, ...
    "F1",100*f1, ...
    "AUC",auc, ...
    "GeometricMean",100*gm, ...
    "MCC",mcc, ...
    "AccuracyCI_Lower",100*ci(1), ...
    "AccuracyCI_Upper",100*ci(2));

M.confusion = struct("TP",tp,"TN",tn,"FP",fp,"FN",fn);
end
