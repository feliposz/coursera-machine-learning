X = zeros(100, 3);
c = 1;
for i = -5:4
    for j = -5:4
        X(c, :) = [i, j, (i+j)/2];
        c = c + 1;
    end
end

X = X + rand(size(X));

x = X(:, 1);
y = X(:, 2);
z = X(:, 3);
figure(1);
plot3(x, y, z, '.');

% covariance matrix
m = size(x, 1);
Sigma = 1/m * X' * X;

% eigenvectors / svd = single value decomposition
[U, S, V] = svd(Sigma);

Ureduce = U(:, 1:2);

Z = X * Ureduce;

x = Z(:, 1);
y = Z(:, 2);

figure(2);
plot3(x, y, zeros(size(x, 1), 1), '.');

Xaprox = Z * Ureduce';

x = Xaprox(:, 1);
y = Xaprox(:, 2);
z = Xaprox(:, 3);

figure(3);
plot3(x, y, z, '.');
