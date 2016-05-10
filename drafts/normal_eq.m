%https://www.coursera.org/learn/machine-learning/lecture/2DKxQ/normal-equation

% Design matrix
X = [ 
      1,2104,5,1,45;
      1,1416,3,2,40;
      1,1534,3,2,30;
      1,852,2,1,36
    ];

% Dimensional vector
y = [460;232;315;178];

% Better when n is small
t = pinv(X' * X) * X' * y
