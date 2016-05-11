function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%t0 = theta(1);
%t1 = theta(2);
%errors = 0;
%for i = 1:m
%  x0 = X(i, 1);
%  x1 = X(i, 2);
%  prediction = t0*x0 + t1*x1;
%  errors = errors + (prediction - y(i))^2;
%end
%J = 1 / (2*m) * errors;

J = 1 / (2*m) * sum((X * theta - y) .^ 2);

% =========================================================================

end
