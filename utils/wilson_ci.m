function ci = wilson_ci(k,n,level)
%WILSON_CI Wilson score interval for a binomial proportion.
if nargin<3, level=0.95; end
if n==0, ci=[NaN NaN]; return; end
alpha=1-level;
z=norminv(1-alpha/2);
p=k/n;
den=1+z^2/n;
ctr=(p+z^2/(2*n))/den;
half=z*sqrt((p*(1-p)/n)+z^2/(4*n^2))/den;
ci=[max(0,ctr-half),min(1,ctr+half)];
end
