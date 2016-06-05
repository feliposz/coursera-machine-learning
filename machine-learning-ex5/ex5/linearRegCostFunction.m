function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X     12x2
% y     12x1
% theta 2x1

h = X * theta; %12x1

s = (h - y)' * (h - y); %1x12 * 12x1 = 1x1

t = theta;
t(1) = 0;
s2 = t' * t; % 1x2 * 2x1 = 1*1

J = 1/(2*m) * s + lambda/(2*m) * s2; %1x1

grad = 1/m * ((h - y)' * X)'; %1x12 * 12x2 = 1x2 -> ' -> 2 x 1
grad = grad + (lambda/m * t); % 2 x 1 + 2 x 1

% =========================================================================

grad = grad(:);

end
