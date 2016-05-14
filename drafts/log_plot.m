x = 0:0.001:1;
y0 = -log(1-x);
y1 = -log(x);
figure(1);
plot(x,y0,x,y1);
axis tight;
title('-log');
legend('y0 = -log(1-x)', 'y1 = -log(x)');

y0 = log(1-x);
y1 = log(x);
figure(2);
plot(x,y0,x,y1);
axis tight;
title('log');
legend('y0 = log(1-x)', 'y1 = log(x)');
