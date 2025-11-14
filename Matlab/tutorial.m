% Author: Scott Albert
% Email: salbert8@jhu.edu
% Institution: Johns Hopkins University
% Lab: Laboratory for Computational Motor Control
% Advisor: Reza Shadmehr
% Date: July 25, 2017
% Location: Baltimore, MD 21211
% % Version: 1.1

% Summary: This MATLAB script demonstrates how to use EM to fit a
% two state model of adaptation to a sensorimotor task that includes
% error-clamp trials. First, behavior is simulated according to a
% two state model of learning. Second, EM is used to fit the behavior.

% Notes: For more information about this package see README.pdf.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PART 1: The Paradigm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here we specify the experimental paradigm. The paradigm is defined by the
% sequence of error-clamp trials and perturbations. If trial n
% is not marked as any of these trials, it is assumed that trial n is a
% trial where the subject is unperturbed.

% the perturbation schedule
%     a value of zero indicates no perturbation
%     a NaN value indicates an error-clamp trial
%     a non-zero value indicates a non-zero perturbation
r = [zeros(20,1) ; 30*ones(50,1) ; nan(20,1) ; zeros(30,1)];

% the error-clamp sequence
%     a value of zero indicates this trial is not an error-clamp trial
%     a non-zero entry indicates that this trial is an error-clamp trial
EC = [zeros(20,1)  ; zeros(50,1) ; ones(20,1) ; zeros(30,1)];

% the value of the error during error-clamp trials
%     a NaN value indicates that this trial was not an error-clamp trial
%     a numeric value indicates the clamped error on this trial
EC_value = [nan(20,1)  ; nan(50,1) ; zeros(20,1) ; nan(30,1)];

% gets the number of trials
N = length(r);

% specifies the c parameter
%     should be equal to [1 ; 1] if it is assumed that behavior is
%     determined by the unweighted sum of the fast and slow states
c = [1;1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% PART 2: Simulation of behavior %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify a two-state model parameter set for simulation
aS = 0.985; % the slow state retention factor
aF = 0.556; % the fast state retention factor
bS = 0.097; % the slow state error sensitivity
bF = 0.213; % the fast state error sensitivity
xS1 = 0; % the initial slow state
xF1 = 0; % the initial fast state
sigmax2 = 1.694; % variance of the update of the slow state
sigmau2 = 1.037; % variance in the execution of a movement
sigma12 = 0; % variance in the initial state

% construct a parameter set from the two state model parameters
simulation_parameters = [aS;aF;bS;bF;xS1;xF1;sigmax2;sigmau2;sigma12];

% seed the random number generator to get the same simulation results
% across different runs of this script
rng(5)

% calls a separate function to simulate behavior corresponding to the
% two state model parameter set and the paradigm
[y,xS,xF] = two_state_simulation_with_noise(simulation_parameters,r,EC,EC_value,c);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% PART 3: Fit two state model with EM %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specifies the number of EM iterations
num_iterations = 100;
 
% determines if a mex function will be used for the model fit
use_mex = 1; % if non-zero, a mex function is used to compute likelihood

% set the boundaries for the parameter search
% each row specifies the min,max pair for a two state model parameter
% each row corresponds to the parameter order specified in PART 2
search_space = [0,1.1 ; ... % bounds for the slow state retention factor
                0,1.1 ; ... % bounds for the fast state retention factor
                0,1 ; ... % bounds for the slow state error sensitivity
                0,1 ; ... % bounds for the fast state error sensitivity
                -30,30 ; ... % bounds for the initial slow state
                -30,30 ; ... % bounds for the initial fast state
                0.0000001,10 ; ... % bounds for state noise variance
                0.0000001,10  ; ... % bounds for motor variance
                0.0000001,10]; % bounds for initial state variance

% specifies inequalities relating the parameter values to enforce two state
% model dynamics
constraints = zeros(2,1);
% specifies deltaA for the inequality: aSlow >= deltaA + aFast
constraints(1) = 0.001; 
% specifies deltaB for the inequality: bFast >= deltaB + bSlow
constraints(2) = 0.001;

% specify an initial parameter guess for the EM algorithm
IC_EM = [0.95, ... % initial guess of slow state retention factor
         0.7, ... % initial guess of fast state retention factor
         0.05, ... % initial guess for the slow state error sensitivity
         0.30, ... % initial guess for the fast state error sensitivity
         0, ... % initial guess for the initial slow state
         0, ... % initial guess for the initial fast state
         2, ... % initial guess for the state update variance
         2, ... % initial guess for the motor variance
         5]; % initial guess for the initial state variance

% use the EM algorithm to fit the two state model
[parameters_EM,likelihoods_EM] = generalized_expectation_maximization(...
    IC_EM,y,r,EC,EC_value,c,search_space,constraints,...
    num_iterations,use_mex);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% PART 4: Visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% use the EM parameters to simulate the behavior, fast, and slow states
[y_EM,xS_EM,xF_EM] = two_state_simulation_without_noise(parameters_EM,r,EC,EC_value,c);

% plot the paradigm, the simulated behavior, and the EM fit
figure
% behavior
subplot(3,1,1)
hold on
plot(1:N,EC,'b')
plot(1:N,r,'r')
plot(1:N,y,'-k')
plot(1:N,y_EM,'--k')
xlim([1,N])
legend('error-clamp','perturbation','motor ouput','EM')
xlabel('Trial number')
ylabel('Motor output')
% slow state
subplot(3,1,2)
hold on
plot(1:N,xS,'-k')
plot(1:N,xS_EM,'--k')
xlim([1,N])
legend('Truth','EM')
xlabel('Trial number')
ylabel('Slow state of learning')
% fast state
subplot(3,1,3)
hold on
plot(1:N,xF,'-k')
plot(1:N,xF_EM,'--k')
xlim([1,N])
legend('Truth','EM')
xlabel('Trial number')
ylabel('Fast state of learning')

% plot the incomplete log-likelihood as a function of EM iteration
figure
plot(likelihoods_EM,'-k')
xlim([1,num_iterations])
xlabel('EM iteration')
ylabel('log[L(y|\theta)]')