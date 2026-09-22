function export_environment(outFile)
%EXPORT_ENVIRONMENT Save MATLAB/toolbox environment for archival reproducibility.
if nargin<1, outFile="environment.txt"; end
v=ver;
fid=fopen(outFile,"w");
fprintf(fid,"MATLAB version: %s\n",version);
fprintf(fid,"Computer: %s\n\n",computer);
for i=1:numel(v)
    fprintf(fid,"%s | %s | %s\n",v(i).Name,v(i).Version,v(i).Release);
end
fclose(fid);
end
