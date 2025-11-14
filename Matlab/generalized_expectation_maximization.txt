function [parameters,likelihoods] = generalized_expectation_maximization(...
    parameters,y,r,EC,EC_value,c,search_space,constraints,...
    num_iterations,use_mex)
% Author: Scott Albert
% Email: salbert8@jhu.edu
% Institution: Johns Hopkins University
% Lab: Laboratory for Computational Motor Control
% Advisor: Reza Shadmehr
% Date: July 25, 2017
% Location: Baltimore, MD 21211
% % Version: 1.1
%
% Summary: This function coordinates the EM algorithm. It successively
%     calls the Kalman Smoother to specify the E-Step, and then performs
%     the M-step to update the parameter set.
%
% Notes: For more information about this package see README.pdf.
%
% Input description:
%    parameters: an initial guess of the two state model parameters that
%        seed the EM algorithm
%    y: the motor output on each trial
%    r: the perturbation on each trial
%    EC: an array that indicates if a trial is an error-clamp trial
%        If the n-th entry is non-zero, this indicates that trial n is an
%        error-clamp trial
%        If the n-th entry is zero, this indicates that trial n is not an
%        error-clamp trial
%    EC_value: an array that indicates the value of the clamped error on
%        each error-clamp trial. 
%    c: a model parameter that is assumed invariant
%    search_space: a matrix containing upper and lower bounds for the model
%        parameters
%    constraints: an array that specifies linear inequality constraints
%        between the fast and slow retention factors, and error
%        sensitivities
%    num_iterations: the number of EM iterations
%    use_mex: determines if a mex function will be used in the M-step of
%        the algorithm, or a regular MATLAB function (m-file)
%
% Output description:
%    parameters: the final parameters obtained at the conclusion of all the
%        EM iterations
%    likelihoods: an array containing the value of the incomplete
%        log-likelihood function on each iteration of the EM algorithm

% computes the error on each trial
N = length(y); % the number of trials
e = zeros(N,1); % allocates space for the errors
for n = 1 : N % iterate through each trial
    if EC(n) == 0
        % this is not an error-clamp trial
        e(n) = r(n) - y(n);
    else
        % this is an error-clamp trial
        e(n) = EC_value(n);
    end
end

% creates an array that stores the value of the incomplete log-likelihood
% function on each iteration of the EM algorithm
likelihoods = zeros(num_iterations,1);

% the EM algorithm
for n = 1 : num_iterations
    % E-step:
    %    get the smoothed Kalman estimates of states, variances, and
    %    covariances using a Kalman smoother
    [xnN,VnN,Vnp1nN] = kalman_smoother(parameters,y,e,c);
    
    % M-step:
    %    perform maximum likelihood estimation in a contrained parameter
    %    space
    parameters = m_step(parameters,xnN,VnN,Vnp1nN,y,e,c,search_space,...
        constraints,use_mex);
    
    % computes the incomplete log-likelihood for this parameter set
    likelihoods(n) = incomplete_log_likelihood(y,e,c,parameters);
    
    % checks to make sure that the likelihood function has increased
    if (n > 1) && (likelihoods(n) < likelihoods(n-1))
        % the likelihood has not increased, warn the modeler
        warning('The expected complete log-likelihood function has stopped increasing');
    end
end