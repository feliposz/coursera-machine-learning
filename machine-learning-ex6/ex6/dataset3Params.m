function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%C_list = [0.1, 1, 10];
%sigma_list = [0.1, 1, 10];

Error = zeros(length(C_list), length(sigma_list));

for i=1:length(C_list)
  for j=1:length(sigma_list)
    fprintf('C: %f sigma: %f\n', C_list(i), sigma_list(j));
    model = svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(j)));
    predictions = svmPredict(model, Xval);
    Error(i, j) = mean(double(predictions ~= yval));
  end
end

[e, ie] = min(Error(:));

fprintf('e: %f, ie: %d\n', e, ie);

i = mod((ie-1), size(Error, 1)) + 1;
j = floor((ie-1) / size(Error, 1)) + 1;

C = C_list(i);
sigma = C_list(j);

fprintf('best -> C: %f sigma: %f\n', C, sigma);

% =========================================================================

end
