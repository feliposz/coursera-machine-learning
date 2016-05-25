y = ones(3, 3, 3)
y(:,:,1) = magic(3);
y(:,:,2) = magic(3) .^ 2;
y(:,:,3) = magic(3) .^ 3;


r = 0;
for i = 1:m
    for k = 1:K
        h = sigmoid(Theta(:, :,  k) * X);
        r = r + y(:,k,i) * log(h) + (1-y(:,k,i)) * log(1-h);
    end
end

Temp = Theta;
Temp(:,:,1) = 0; % ???
r = r + lambda/m * sum(Temp .^ 2);

J = 1/m * r;

