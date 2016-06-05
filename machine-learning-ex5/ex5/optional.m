%% Initialization
clear ; close all; clc

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);


Xall = [X; Xtest; Xval];
yall = [y; ytest; yval];

% m = Number of examples
m = size(Xall, 1);

% Plot training data
%plot(Xall, yall, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
%xlabel('Change in water level (x)');
%ylabel('Water flowing out of the dam (y)');


iterations = 3;

num_train = round(m * 0.2);
num_val = round(m * 0.2);

error_train_sum = 0;
error_val_sum = 0;

for iter = 1:iterations

  % random selection
  i_train = randperm(m, num_train);
  i_val = randperm(m, num_val);
  Xtrain = Xall(i_train);
  ytrain = yall(i_train);
  Xval = Xall(i_val);
  yval = yall(i_val);
  

  p = 8;

  X_poly_train = polyFeatures(Xtrain, p);
  [X_poly_train, mu, sigma] = featureNormalize(X_poly_train);
  X_poly_train = [ones(size(X_poly_train, 1), 1), X_poly_train];

  % Map X_poly_val and normalize (using mu and sigma)
  X_poly_val = polyFeatures(Xval, p);
  X_poly_val = bsxfun(@minus, X_poly_val, mu);
  X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
  X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];
  
  lambda = 0.01;
  [error_train, error_val] = learningCurve(X_poly_train, ytrain, X_poly_val, yval, lambda);
  
  error_train_sum = error_train_sum  + error_train;
  error_val_sum = error_val_sum  + error_val;
  
end

error_train_avg = 1/iterations * error_train_sum;
error_val_avg = 1/iterations * error_val_sum;

plot(1:length(error_train_avg), error_train_avg, 1:length(error_val_avg), error_val_avg);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
