X = linspace(-pi, pi);
figure(4);
plot(X, cos(X), 'r');
hold on;
plot(X, sin(X), 'b');
legend('cos(x)', 'sin(x)');