function T = parameter_audit(cfg)
%PARAMETER_AUDIT Analytic trainable-parameter audit for the documented graph.
%
% With maximal grouping and grouped Negative-Down, the total is expected to
% be approximately 6.364 M, matching the manuscript's rounded 6.4 M value.

W=cfg.Model.Widths;
total=0;
parts={};

% Stem
pStem = convp(7,3,W(1)) + convp(4,3,W(1)) + bnLearn(W(1));
total=total+pStem; parts(end+1,:)={"Negative-Stem",pStem}; %#ok<AGROW>

for s=1:4
    F=W(s);
    p = 0;
    p = p + gconvp(3,F,F);       % ten1
    p = p + gconvp(1,F,F);       % ten3
    p = p + bnLearn(2*F);        % ten8 BN
    p = p + gconvp(1,2*F,4*F);   % ten9 expansion
    p = p + convp(1,4*F,F);      % ten9 projection
    p = p + gconvp(1,F,4*F);     % ten10 expansion
    p = p + convp(1,4*F,F);      % ten10 projection
    p = p + bnLearn(F);           % ten11 BN
    total=total+p;
    parts(end+1,:)={"NegativeKAN "+s,p}; %#ok<AGROW>

    if s<4
        Fin=W(s); Fout=W(s+1);
        pd = bnLearn(Fin);
        if cfg.Model.UseGroupedDownsample
            pd = pd + gconvp(3,Fin,Fout) + gconvp(2,Fin,Fout);
        else
            pd = pd + convp(3,Fin,Fout) + convp(2,Fin,Fout);
        end
        pd = pd + bnLearn(Fout);
        total=total+pd;
        parts(end+1,:)={"Negative-Down "+s,pd}; %#ok<AGROW>
    end
end

pHead=convp(1,W(end),cfg.Model.NumClasses); % equivalent FC parameter count
total=total+pHead; parts(end+1,:)={"Classifier",pHead}; %#ok<AGROW>

T=cell2table(parts,"VariableNames",["Component","TrainableParameters"]);
T=[T; {"TOTAL",total}];

fprintf("Analytic trainable parameter count: %,d (%.4f M)\n",total,total/1e6);
end

function p=convp(k,cin,cout)
p=k*k*cin*cout+cout;
end
function p=gconvp(k,cin,cout)
g=gcd(cin,cout);
p=k*k*cin*cout/g+cout;
end
function p=bnLearn(c)
p=2*c; % scale + offset
end
