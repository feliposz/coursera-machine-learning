clear;
clc;

% original data
X = (1:0.25:10)';
y = X > 5;

figure(1);
plot(X, y, 'rx');
title('original');

% adding x0 and adapting y
X = [ones(length(X), 1), X];

theta = [0; 0];

function [jVal, gradient] = costFunction(theta, X, y)
  m = length(y);
  z = X*theta;
  h = 1 ./ (1 + exp(-z));
  jVal = (1/m) * sum(-y.*log(h) - (1-y).*log(1-h));
  gradient = (1/m) * ((h - y)' * X)';
end


options = optimset('GradObj', 'on', 'MaxIter', '1000');
initialTheta = zeros(2, 1);
%fminunc - function minimal unconstrained
[optTheta, functionVal, exitFlag] = fminunc(@(t)costFunction(t, X, y), initialTheta, options)

x2 = X(:,2);
y2 = 1 ./ (1 + exp(-X * optTheta));
figure(1);
hold('on');
plot(x2, y2, 'b-');
hold('off');

%TODO: fix this!!!

