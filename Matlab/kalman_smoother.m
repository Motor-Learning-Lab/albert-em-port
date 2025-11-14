function [xnN,VnN,Vnp1nN] = kalman_smoother(parameters,y,e,c)
% Author: Scott Albert
% Email: salbert8@jhu.edu
% Institution: Johns Hopkins University
% Lab: Laboratory for Computational Motor Control
% Advisor: Reza Shadmehr
% Date: July 25, 2017
% Location: Baltimore, MD 21211
% % Version: 1.1
%
% Summary: This function implements the Kalman smoother. It computes the
%    expected value of the state and covariances, given a set of
%    behavioral observations and the two state model parameters.
%
% Notes: For more information about this package see README.pdf.
%
% Input description:
%    parameters: the current estimate of the two-state model parameters
%    y: the motor output on each trial
%    e: the error experienced by the subject on each trial
%    c: a model parameter that is assumed invariant
%
% Output description:
%    xnN: This is shorthand for the quantity x(n|N). It is the smoothed
%        Kalman state expectation  E[x(n)|y(1),y(2),...,y(N)].
%    VnN: This is shorthand for the quantity V(n|N). It is the smoothed
%        Kalman state variance  var(x(n)|y(1),y(2),...,y(N)).
%    Vnp1nN: This is shorthand for the quantity V(n+1,n|N). It is the
%        smoothed Kalman covariance of consecutive states, also written as
%        cov(x(n+1),x(n)|y(1),y(2),...,y(N)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% stores parameter values %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kalman smoother %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this section, a Kalman smoother is used to compute the smoothed
% expectations and covariances of the hidden states.
%
% Notation:
%    N it the total number of trials
%    the smoothed expectation E(x(n)|y(1),...,y(N)) is denoted xnN
%    the smoothed variance var(x(n)|y(1),...,y(N)) is denoted VnN
%    the smoothed covariance cov(x(n+1),x(n)|y(1),...,y(N)) is denoted 
%       Vnp1nN

% allocate space for the smoothed expectations and variances
xnN = cell(N,1);
VnN = cell(N,1);

% instantiate the expectation and variance of the final trial as the
% posteriors obtained at the end of the forward Kalman filter
xnN{end} = xnn{end};
VnN{end} = Vnn{end};

% allocate space for the J parameter
Jn = cell(N,1);

% backwards recursions for Kalman smoothing
for n = N - 1 : -1 : 1
    % computes J
    Jn{n} = Vnn{n}*((A')/Vnnm1{n + 1});
    
    % computes the smoothed variance
    VnN{n} = Vnn{n} + Jn{n}*(VnN{n+1} - Vnnm1{n+1})*(Jn{n}');    
    
    % computes the smoothed expectation
    xnN{n} = xnn{n} + Jn{n}*(xnN{n+1} - xnnm1{n+1});
end

% computes the smoothed covariances
Vnp1nN = cell(N,1);
for n = 1 : N - 1
    Vnp1nN{n} = VnN{n+1}*(Jn{n}');
end