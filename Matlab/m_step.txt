function [parameters_final] = m_step(parameters_0,xnN,VnN,Vnp1nN,y,e,c,...
    search_space,constraints,use_mex)
% Author: Scott Albert
% Email: salbert8@jhu.edu
% Institution: Johns Hopkins University
% Lab: Laboratory for Computational Motor Control
% Advisor: Reza Shadmehr
% Date: July 25, 2017
% Location: Baltimore, MD 21211
% % Version: 1.1
%
% Summary: This function uses fmincon to perform the M-step of the EM
%    algorithm. The parameter set that maximizes the expected complete
%    log-likelihood function in a constrained parameter space is identified
%    using fmincon.
%
% Notes: For more information about this package see README.pdf.
%
% Input description:
%    parameters_0: the current estimate of the two state model parameters
%    xnN: This is shorthand for the quantity x(n|N). It is the smoothed
%        Kalman state expectation  E[x(n)|y(1),y(2),...,y(N)].
%    VnN: This is shorthand for the quantity V(n|N). It is the smoothed
%        Kalman state variance  var(x(n)|y(1),y(2),...,y(N)).
%    Vnp1nN: This is shorthand for the quantity V(n+1,n|N). It is the
%        smoothed Kalman covariance of consecutive states, also written as
%        cov(x(n+1),x(n)|y(1),y(2),...,y(N)).
%    y: the motor output on each trial
%    e: the error experienced by the subject on each trial
%    c: a model parameter that is assumed invariant
%    search_space: a matrix containing upper and lower bounds for the model
%        parameters
%    constraints: an array that specifies linear inequality constraints
%        between the fast and slow retention factors, and error
%        sensitivities
%    use_mex: determines if an mex function will be used or a regular
%        MATLAB function (m-file)
%
% Output description:
%    parameters_final: the parameter set identified by fmincon that
%        maximizes the expected complete log-likelihood function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Call to the likelihood function %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% creates a local function to be used by fmincon that computed the negated
% value of the expected complete log-likelihood function
function negated_likelihood = likelihood_function(x)
    if use_mex
        % calls an executable mex function to compute the likelihood
        likelihood = expected_complete_log_likelihood_mex(x,y,e,c,xnN,VnN,Vnp1nN);
    else
        % calls a MATLAB function to compute the likelihood
        likelihood = expected_complete_log_likelihood(x,y,e,c,xnN,VnN,Vnp1nN);
    end
        
    % negates the likelihood
    negated_likelihood = -likelihood;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% set up and perform fmincon %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% resets options for fmincon
options = optimoptions('fmincon');
options.Display = 'off';

% specifies the lower and upper bounds for the fmincon search
lb = search_space(:,1);
ub = search_space(:,2);

% specifies linear constraints for the fmincon search
%        aS    aF   bS   bF   xS1   xF1   sigmax2    sigmau2  sigma12
A_con = [-1,   1,   0,   0,   0,    0,      0,          0,       0;
          0,   0,   1,  -1,   0,    0,      0,          0,       0];
b_con = [-constraints(1);  % -a_slow + a_fast < -C
         -constraints(2)];  % b_slow - b_fast < -C

% uses fmincon to maximize the expected complete log-likelihood function in
% a constrained parameter space
parameters_final = fmincon(@likelihood_function, parameters_0, A_con, b_con, ...
    [], [], lb, ub, [], options);
end