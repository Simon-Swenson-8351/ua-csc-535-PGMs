% Q3.a
figure(1);
% (2, 0.2)
X = linspace(1, 3);
plot(X, normpdf(X, 2, 0.2), 'r');
hold on;
% (1, 0.5)
X2 = linspace(-1.5, 3.5);
plot(X2, normpdf(X2, 1, 0.5), 'g');
% (0, 2)
X3 = linspace(-10, 10);
plot(X3, normpdf(X3, 0, 2), 'b');
legend('mu = 2, var = 0.2', 'mu = 1, var = 0.5', 'mu = 0, var = 2');

% Q3.b
figure(2);
multivarGaussMin = -5;
multivarGaussMax = 5;
multivarGaussStep = 0.1;
[X, Y] = meshgrid(multivarGaussMin:multivarGaussStep:multivarGaussMax, multivarGaussMin:multivarGaussStep:multivarGaussMax);
Z = reshape(mvnpdf([X(:) Y(:)], [0, 0], [0.5 0.3; 0.3 2.0]), length(X), length(Y));
surf(X, Y, Z);

% Q4
% For a given x value, estimate p(x) by adding up all rectangular prisms
% with that same x value.
marginalized = sum(Z, 1) * multivarGaussStep * multivarGaussStep;
figure(3);
plot(X(1, :), marginalized);

% Q5
% I hard-code the column where X = 0.5 here. Bad but meh.
figure(4)
plot(Y(:, 56), Z(:, 56)/sum(Z(:, 56)));