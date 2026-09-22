function Xs = apply_cwinca(X,model)
%APPLY_CWINCA Apply training-fitted min-max normalization and fixed indices.
X=double(X);
Xn=(X-model.xmin)./(model.xmax-model.xmin+model.epsilon);
Xs=Xn(:,model.indices);
end
