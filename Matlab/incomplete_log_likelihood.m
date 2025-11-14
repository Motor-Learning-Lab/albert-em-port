function [likelihood] = incomplete_log_likelihood(y,e,c,parameters)
% Author: Scott Albert
% Email: salbert8@jhu.edu
% Institution: Johns Hopkins University
% Lab: Laboratory for Computational Motor Control
% Advisor: Reza Shadmehr
% Date: July 25, 2017
% Location: Baltimore, MD 21211
% % Version: 1.1
%
% Summary: This function computes the incomplete log-likelihood function.
%    The EM algorithm attempts to increase the value of this function each
%    iteration. It is the function maximized in standard MLE.
%
% Notes: For more information about this package see README.pdf.
%
% Input description:
%    y: the motor output on each trial
%    e: the error experienced by the subject on each trial
%    c: a model parameter that is assumed invariant
%    parameters: a two state model parameter set
%
% Output description:
%    likelihood: the incomplete log-likelihood for this parameter set

% stores input variables using descriptive names
aS = parameters(1);
aF = parameters(2);
bS = parameters(3);
bF = parameters(4);
xS1 = parameters(5);
xF1 = parameters(6);
sigmax2 = parameters(7);
sigmau2 = parameters(8);
sigma12 = parameters(9);

% sets the means and variances for the initial states
x1 = [xS1;xF1];
V1 = [sigma12, 0; 0, sigma12];

% sets matrices and vectors for the update of the fast and slow states
b = [bS;bF];
A = [aS,0;0,aF];
Q = [sigmax2,0;0,sigmax2];

% determines the number of trials
N = length(y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% forward Kalman filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this section, a forward Kalman filter is used to compute prior and
% posterior expectations and variances of the state x.
%
% Notation:
%    the posterior expectation E(x(n)|y(1),...,y(n)) is denoted xnn
%    the posterior variance var(x(n)|y(1),...,y(n)) is denoted Vnn
%    the prior expectation E(x(n)|y(1),...,y(n-1)) is denoted xnnm1
%    the prior variance var(x(n)|y(1),...,y(n-1)) is denoted Vnnm1

% allocates space for arrays for the prior and posterior expectations and
% variances of the hidden states
xnnm1 = cell(N,1);
Vnnm1 = cell(N,1);
xnn = cell(N,1);
Vnn = cell(N,1);

% specifies the initial prior, x(1|0) = x1
xnnm1{1} = x1;

% specifies the initial prior variance, V(1|0) = V1
Vnnm1{1} = V1;

% the standard forward Kalman filter
for n = 1 : N    
    % computes the Kalman gain
    k = (Vnnm1{n}*c) / ((c')*Vnnm1{n}*c + sigmau2);
    
    % computes the error between our actual and predicted y values
    y_error = y(n) - (c')*xnnm1{n};
    
    % computes the posterior state expectation
    xnn{n} = xnnm1{n} + k*y_error;
    
    % computes the posterior state variance
    Vnn{n} = (eye(2) - k*(c'))*Vnnm1{n};    
    
    % forward projects, unless the last trial has been reached
    if n < N        
        % computes the next prior state
        xnnm1{n+1} = A*xnn{n} + b*e(n);
        
        % computes the next prior variance
        Vnnm1{n+1} = A*Vnn{n}*(A') + Q;
    end
end

% computes the log-likelihood, log[L(y(1),y(2),...y(N)|parameters)]
likelihood = -(N/2)*log(2*pi);
for n = 1 : N
    % the variance and mean of the normal random variable
    SIGMA = (c')*Vnnm1{n}*c + sigmau2;
    MU = (c')*xnnm1{n};
    
    % updates the likelihood
    likelihood = likelihood - (1/2)*log(SIGMA) - ...
        (1/2) * ( (y(n) - MU)^2 ) / SIGMA;
end