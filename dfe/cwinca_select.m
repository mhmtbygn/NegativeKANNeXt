function model = cwinca_select(X,y,cfg)
%CWINCA_SELECT NCA-guided cumulative-weight iterative subset selection.
%
% Training-only fitting. Test data must use model normalization and indices.

X = double(X);
y = categorical(y);

xmin = min(X,[],1);
xmax = max(X,[],1);
epsv = 1e-12;
Xn = (X-xmin)./(xmax-xmin+epsv);

nca = fscnca(Xn,y,"Standardize",false);
w = nca.FeatureWeights(:);
[ws,order] = sort(w,"descend");

if sum(ws)<=0
    ws = ones(size(ws));
end
cw = cumsum(ws)/sum(ws);

kStart = find(cw>=cfg.DFE.LowerThreshold,1,"first");
kStop  = find(cw>=cfg.DFE.UpperThreshold,1,"first");
if isempty(kStart), kStart=1; end
if isempty(kStop), kStop=size(X,2); end
kStart=max(1,kStart);
kStop=max(kStart,kStop);

cvp = cvpartition(y,"KFold",cfg.DFE.InnerFolds);
losses = inf(kStop-kStart+1,1);

for ii=1:numel(losses)
    k=kStart+ii-1;
    Xi=Xn(:,order(1:k));
    foldLoss=zeros(cvp.NumTestSets,1);
    for f=1:cvp.NumTestSets
        tr=training(cvp,f); te=test(cvp,f);
        mdl=fitcknn(Xi(tr,:),y(tr), ...
            "NumNeighbors",1,"Distance","cityblock", ...
            "DistanceWeight","equal","Standardize",true);
        p=predict(mdl,Xi(te,:));
        foldLoss(f)=mean(p~=y(te));
    end
    losses(ii)=mean(foldLoss);
end

[bestLoss,bestPos]=min(losses);
bestK=kStart+bestPos-1;
indices=order(1:bestK);

model.indices = indices(:)';
model.bestK = bestK;
model.bestLoss = bestLoss;
model.weights = w;
model.sortedOrder = order;
model.cumulativeWeights = cw;
model.xmin = xmin;
model.xmax = xmax;
model.epsilon = epsv;
model.searchK = (kStart:kStop)';
model.searchLoss = losses;
end
