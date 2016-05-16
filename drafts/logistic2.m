clear;
clc;

% original data
X = 1:10;
y(1:5) = 0;
y(6:10) = 1;

figure(1);
plot(X, y, 'rx');
title('original');

% adding x0 and adapting y
X = [ones(length(X), 1), X'];
y = y';

theta = zeros(2, 1);

iterations = 1000;
alpha = 0.1;
m = length(y);

for i = 1:iterations

  z = X * theta;
  h = 1 ./ (1 + exp(-z));

  theta = theta - alpha * (1/m) * ((h - y)' * X)';

  J = (1/m) * sum(-y.*log(h) - (1-y).*log(1-h));
  J_history(i) = J;
  
end

figure(2);
plot(1:iterations, J_history);
title('Cost history');

x2 = X(:,2);
y2 = 1 ./ (1 + exp(-X * theta));
figure(1);
hold('on');
plot(x2, y2, 'b-');
hold('off');

theta
