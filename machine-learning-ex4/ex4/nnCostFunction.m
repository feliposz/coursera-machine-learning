function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1 - Cost function

% Theta1  25 x 401
% Theta2  10 x 26
% X       5000 x 400
% y       5000 x 1

Y = zeros(m, num_labels); % 5000 x 10

for i = 1:m
  j = y(i);
  Y(i, j) = 1;
end

Z = sigmoid([ones(size(X,1),1) X] * Theta1'); % 5000 x 25
H = sigmoid([ones(size(Z,1),1) Z] * Theta2'); % 5000 x 10

T = -Y .* log(H) - (1-Y) .* log(1-H); % 5000 x 10

% Regularization without the 0th element
RegTheta1 = Theta1(:, 2:end); % 25 x 400
RegTheta2 = Theta2(:, 2:end); % 10 x 25
Reg = [RegTheta1(:) ; RegTheta2(:)]; % 10250 x 1

J = 1/m * sum(sum(T)) + lambda/(2*m) * Reg' * Reg; % 1 x 1


% Part 2 - Backpropagation and derivatives

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

do_vectorization = true;

if do_vectorization

  % 17s for 50 iterations

  % Forward propagation
  a_1 = X;
  a_1 = [ones(size(X,1),1) a_1]; % 5000 x 401
  z_2 = a_1 * Theta1'; % 5000 x 25
  a_2 = sigmoid(z_2); % 5000 x 25
  a_2 = [ones(size(a_2,1),1) a_2]; % 5000 x 26
  z_3 = a_2 * Theta2'; % 5000 x 10
  a_3 = sigmoid(z_3); % 5000 x 10

  % Backpropagation
  delta_3 = a_3 - Y; % 5000 x 10
  delta_2 = delta_3 * Theta2 .* sigmoidGradient([ones(size(z_2,1),1) z_2]); % 5000 x 26
  delta_2 = delta_2(:, 2:end); % remove 0th column => 5000 x 25

  % Accumulate gradients
  D1 = delta_2' * a_1; % 25 x 401
  D2 = delta_3' * a_2; % 10 x 26

else

  % 1m28s for 50 iterations

  for t = 1:m
    
    % Forward propagation
    a_1 = X(t, :)'; % 400 x 1
    a_1 = [1; a_1]; % 401 x 1
    z_2 = Theta1 * a_1; % 25 x 1
    a_2 = sigmoid(z_2); % 25 x 1
    a_2 = [1; a_2]; % 26 x 1
    z_3 = Theta2 * a_2; % 10 x 1
    a_3 = sigmoid(z_3); % 10 x 1
    
    % Backpropagation
    delta_3 = a_3 - Y(t, :)';  % 10 x 1
    delta_2 = Theta2' * delta_3 .* sigmoidGradient([1; z_2]); % 26 x 1
    delta_2 = delta_2(2:end); % remove 0th element => 25 x 1
    
    % Accumulate gradients
    D1 = D1 + delta_2 * a_1'; % 25 x 401
    D2 = D2 + delta_3 * a_2'; % 10 x 26
    
  end
  
end



% Gradients
Theta1_grad = 1/m * D1;
Theta2_grad = 1/m * D2;

% Part 3 - Regularization

T1 = Theta1;
T2 = Theta2;
T1(:, 1) = 0;
T2(:, 1) = 0;

Theta1_grad = Theta1_grad + lambda/m * T1;
Theta2_grad = Theta2_grad + lambda/m * T2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

