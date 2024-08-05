clear;
close all;

%% Add GMR lib and 'C' shape (from LASA dataset)
addpath(genpath('lib'))
addpath(genpath('data'))

%% Load data
load('CShape.mat');

%% Create input/output data for GMM-based diffeomorphism
D = 10;
subsampleInd = 1:D:size(demos{1}.pos,2);
k = 1.3;
gmmData = [];
dt = D*dt;
for d=1:3
    x = demos{d}.pos(:,1);
    for t=subsampleInd % size(demos{d}.pos,2)
       dx = -k * x;
       x = x + dx*dt;
       
       gmmData = [gmmData, [x; demos{d}.pos(:,t)]];
    end
end

%% Learn a diffeomorphism
in = 1:2;
out = 3:4;  
nbStates = 30;
% Fit GMM
[Priors, Mu, Sigma] = EM_init_kmeans(gmmData, nbStates);
[Priors, Mu, Sigma] = EM(gmmData, Priors, Mu, Sigma);   
% Create function handles
fun = @(pt) GMR(Priors, Mu, Sigma, pt, in, out); % Diffeomorphism
funReverse = @(pt) GMR(Priors, Mu, Sigma, pt, out, in); % Inverse diffeomorphism
funJac = @(pt) GMR_with_Jacobian(Priors, Mu, Sigma, pt, in, out); % Diffeomorphism and its Jacobian

%% Test the learned mappings
N = 300;
figure(1)
subplot(1,2,1)
plot(gmmData(3,1:N),gmmData(4,1:N),'r.')
hold on
regOut = fun(gmmData(in,1:N));
plot(regOut(1,1:N),regOut(2,1:N),'b.')
title('Direct mapping')

subplot(1,2,2)
plot(gmmData(1,1:N), gmmData(2,1:N),'r.')
hold on
regIn = funReverse(gmmData(out,1:N));
plot(regIn(1,1:N), regIn(2,1:N),'b.')
title('Inverse mapping')