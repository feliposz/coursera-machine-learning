clear;
clc;
x0 = 1;
x1 = -4:0.25:4;
x2 = -4:0.25:4;
t0 = -2;
t1 = 1;
t2 = 1;

for i=1:length(x1)
  for j=1:length(x2)
    z = t0*x0 + t1*x1(i)*x1(i) + t2*x2(j)*x2(j);
    H(i,j) = 1 / (1 + exp(-z));
  end
end

surf(x1, x2, H);
