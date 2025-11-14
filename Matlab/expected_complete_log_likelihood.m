function [likelihood] = expected_complete_log_likelihood(parameters,y,e,...
    c,xnN,VnN,Vnp1nN)
% Author: Scott Albert
% Email: salbert8@jhu.edu
% Institution: Johns Hopkins University
% Lab: Laboratory for Computational Motor Control
% Advisor: Reza Shadmehr
% Date: July 25, 2017
% Location: Baltimore, MD 21211
% % Version: 1.1
%
% Summary: This function computes the expected complete log-likelihood.
%
% Notes: For more information about this package see README.pdf.
% 
% Input description:
%    parameters_0: the current estimate of the two state model parameters
%    y: the motor output on each trial
%    e: the error experienced by the subject on each trial
%    c: a model parameter that is assumed invariant
%    xnN: This is shorthand for the quantity x(n|N). It is the smoothed
%        Kalman state expectation  E[x(n)|y(1),y(2),...,y(N)].
%    VnN: This is shorthand for the quantity V(n|N). It is the smoothed
%        Kalman state variance  var(x(n)|y(1),y(2),...,y(N)).
%    Vnp1nN: This is shorthand for the quantity V(n+1,n|N). It is the
%        smoothed Kalman covariance of consecutive states, also written as
%        cov(x(n+1),x(n)|y(1),y(2),...,y(N)).
% 
% Output description:
%    likelihood: the expected complete log-likelihood

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% allocations and compute matrix products %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sets the values of two-state parameters used to compute the likelhiood
aS = parameters(1);
aF = parameters(2);
bS = parameters(3);
bF = parameters(4);
xS1 = parameters(5);
xF1 = parameters(6);
sigmax2 = parameters(7);
sigmau2 = parameters(8);
sigma12 = parameters(9);

% specifies the mean and variance of the initial states
x1 = [xS1;xF1];
V1 = [sigma12, 0; 0, sigma12];

% stores the number of trials
N = length(y);

% matrices and vectors for the update of the fast and slow states
A = [aS,0;0,aF]; % the A matrix
b = [bS ; bF]; % the b vector
Q = [sigmax2,0;0,sigmax2]; % the Q matrix

% precomputes important quantities referenced in the expected complete
% log-likelihood function
QinvA = Q\A; Qinvb = Q\b; AtQinv = (A')/Q; AtQinvA = AtQinv*A;
AtQinvb = (A')*Qinvb; btQinv = (b')/Q; btQinvA = btQinv*A;
btQinvb = btQinv*b;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% compute the likelihood %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this section the expected complete log-likelihood is computed in
% different parts.

% TERM 1: one of the parts of the expected complete log-likelihood function
% derived from the likelihood of observing the motor output given the
% states
term1 = 0;
for n = 1 : N
    term1 = term1 + y(n)^2 + (c')*(VnN{n} + xnN{n}*(xnN{n}'))*c - ...
        2*y(n)*(c')*xnN{n};
end
term1 = -term1/(2*sigmau2);

% TERM 2: one of the parts of the expected complete log-likelihood function
% derived from the likelihood of observing the state on trial n+1 given the
% state on trial n
term2 = 0;
for n = 1 : N - 1
    term2 = term2 + (xnN{n+1}')*(Q\xnN{n+1}) + trace(Q\VnN{n+1}) - ...
        (xnN{n+1}')*QinvA*xnN{n} - trace(QinvA*(Vnp1nN{n}')) - ...
        (xnN{n+1}')*Qinvb*e(n) - (xnN{n}')*AtQinv*xnN{n+1} - ...
        trace(AtQinv*Vnp1nN{n}) + (xnN{n}')*AtQinvA*xnN{n} + ...
        trace(AtQinvA*VnN{n}) + (xnN{n}')*AtQinvb*e(n) - ...
        e(n)*btQinv*xnN{n+1} + e(n)*btQinvA*xnN{n} + e(n)*btQinvb*e(n);
end
term2 = -term2/2;

% TERM 3: one of the parts of the expected complete log-likelihood function
% derived from the likelihood of observing the initial state
term3 = (xnN{1}')*(V1\xnN{1}) + trace(V1\VnN{1}) - (xnN{1}')*(V1\x1) - ...
    (x1')*(V1\xnN{1}) + (x1')*(V1\x1);
term3 = -term3/2;

% TERM 4: one of the parts of the expected complete log-likelihood function
% derived from the pre-exponential factors
term4 = -(1/2)*log(det(V1)) - (N/2)*log(sigmau2) - (3/2)*N*log(2*pi);

% TERM 5: one of the parts of the expected complete log-likelihood function
% derived from the pre-exponential factors of x(n+1) given x(n)
term5 = -(N-1)*log(det(Q))/2;

% computes the likelihood from the sum of all terms
likelihood = term1 + term2 + term3 + term4 + term5;
