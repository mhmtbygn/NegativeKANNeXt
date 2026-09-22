%% Smoke test: graph construction and basic configuration
clear; clc;
addpath(genpath(fileparts(fileparts(mfilename("fullpath")))));

cfg=defaultConfig();
g=buildNegativeKANNeXt(cfg);

fprintf("Layer count: %d\n",numel(g.Layers));
fprintf("Connection count: %d\n",height(g.Connections));

assert(any(string({g.Layers.Name})=="gap"));
assert(any(string({g.Layers.Name})=="classoutput"));

disp("NegativeKANNeXt graph construction smoke test passed.");
