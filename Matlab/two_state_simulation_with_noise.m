function [y,xS,xF] = two_state_simulation_with_noise(parameters,r,EC,EC_value,c)
% Author: Scott Albert
% Email: salbert8@jhu.edu
% Institution: Johns Hopkins University
% Lab: Laboratory for Computational Motor Control
% Advisor: Reza Shadmehr
% Date: July 25, 2017
% Location: Baltimore, MD 21211
% % Version: 1.1
%
% Summary:
%    This function simulates a two state model. The paradigm is
%    determined by a sequence of perturbations. It allows for trials to
%    be error-clamp trials, but does not account for set breaks.
%
% Notes: For more information about this package see README.pdf.
%
% Input description:
%    parameters: the two-state model parameters
%    r: the perturbation on each trial
%    EC: an array that indicates if a trial is an error-clamp trial
%        If the n-th entry is non-zero, this indicates that trial n is an
%        error-clamp trial
%        If the n-th entry is zero, this indicates that trial n is not an
%        error-clamp trial
%    EC_value: an array that indicates the value of the clamped error on
%        each error-clamp trial. 
%    c: a vector [1;1] used to compute the motor output
%
% Output description:
%    y: the simulated behavior
%    xS: the simulated slow state
%    xF: the simulated fast state

% obtains the parameter values from the input
aS = parameters(1);
aF = parameters(2);
bS = parameters(3);
bF = parameters(4);
xS1 = parameters(5);
xF1 = parameters(6);
sigmax2 = parameters(7);
sigmau2 = parameters(8);
sigma12 = parameters(9);

% creates matrices from the input parameters
b = [bS;bF];
A = [aS,0;0,aF];
Q = [sigmax2,0;0,sigmax2];

% determines the number of trials
N = length(r);

% creates a cell array for the states
states = cell(N,1);

% draws the initial state from a normal distribution
states{1} = mvnrnd([xS1;xF1],[sigma12,0;0,sigma12])';

% generates the states and observations
y = zeros(N,1);

% gets the first simulated y value
y(1) = (c')*states{1} + normrnd(0,sigmau2^0.5);

% simulates all remaining trials
for n = 2 : N
    % determines if this trial is an error-clamp trial
    if EC(n-1) == 0
        % this is not an error-clamp trial
        e = r(n-1) - y(n-1);
    else
        % this is an error-clamp trial
        e = EC_value(n-1);
    end
    
    % updates the state
    states{n} = A*states{n-1} + b*e + mvnrnd([0,0],Q)';
    
    % generate the motor output
    y(n) = (c')*states{n} + normrnd(0,sigmau2^0.5);
end

% stores the slow and fast states as arrays for the output
xS = zeros(N,1);
xF = zeros(N,1);
for n = 1 : N
    xS(n) = states{n}(1);
    xF(n) = states{n}(2);
end