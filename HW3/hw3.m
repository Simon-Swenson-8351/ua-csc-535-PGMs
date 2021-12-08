%q1

q1dist = makedist('Normal', 'mu', 0.0, 'sigma', 1.0);
q1pdf = @(x) pdf(q1dist, x);
q1rand = RandStream('mrg32k3a', 'Seed', 42);
sampleGenerator = DistributionSampleGenerator(q1pdf, -10.0, 10.0, 0.01, q1rand);

X = -10.0:0.1:10.0;
Y = q1pdf(X);

figure(1);
samples = arrayfun(@(x) sampleGenerator.sample(), 1:10);
histogram(samples, 'Normalization', 'pdf');
hold on;
plot(X, Y);

figure(2);
samples = arrayfun(@(x) sampleGenerator.sample(), 1:100);
histogram(samples, 'Normalization', 'pdf');
hold on;
plot(X, Y);

figure(3);
samples = arrayfun(@(x) sampleGenerator.sample(), 1:1000);
histogram(samples, 'Normalization', 'pdf');
hold on;
plot(X, Y);

figure(4);
samples = arrayfun(@(x) sampleGenerator.sample(), 1:10000);
histogram(samples, 'Normalization', 'pdf');
hold on;
plot(X, Y);

makeBetaPdf = @(a, b) @(x) gamma(a + b) / (gamma(a) * gamma(b)) * x.^(a - 1) .* (1 - x).^(b - 1);

beta0 = makeBetaPdf(2.0, 5.0);
q1rand2 = RandStream('mrg32k3a', 'Seed', 42);
sampleGenerator2 = DistributionSampleGenerator(beta0, 0.0, 1.0, 0.001, q1rand2);
figure(7);
samples = arrayfun(@(x) sampleGenerator2.sample(), 1:1000);
histogram(samples, 'Normalization', 'pdf');
hold on;
X = 0:0.01:1.0;
plot(X, beta0(X));

%q2

beta1 = makeBetaPdf(3.0, 3.0);
X2 = 0.0:0.01:1.0;
Y2 = beta1(X2);
figure(5);
plot(X2, Y2);

% Use beta-bernoulli updating
figure(6);

plot(X2, Y2, 'DisplayName', 'Prior (alpha = beta = 3)');
hold on;

% 2 tails
beta2 = makeBetaPdf(3.0, 5.0);
Y2 = beta2(X2);
plot(X2, Y2, 'DisplayName', 'Posterior after 2 tails');

% 4 tails
beta2 = makeBetaPdf(3.0, 7.0);
Y2 = beta2(X2);
plot(X2, Y2, 'DisplayName', 'Posterior after 4 tails');

% 6 tails
beta2 = makeBetaPdf(3.0, 9.0);
Y2 = beta2(X2);
plot(X2, Y2, 'DisplayName', 'Posterior after 6 tails');

% 8 tails
beta2 = makeBetaPdf(3.0, 11.0);
Y2 = beta2(X2);
plot(X2, Y2, 'DisplayName', 'Posterior after 8 tails');

% 10 tails
beta2 = makeBetaPdf(3.0, 13.0);
Y2 = beta2(X2);
plot(X2, Y2, 'DisplayName', 'Posterior after 10 tails');

legend;

%q4
q4rand = RandStream('mrg32k3a', 'Seed', 535);
q4X = -1.0:0.2:1.0;
q4polyfn = @(x) x.^3 - x;
q4polysamples = @(x) q4polyfn(x) + q4rand.randn(1, size(x, 2)) .* 0.2;

q4Y = q4polysamples(q4X);
q4YOracle = q4polyfn(q4X);

figure(8);
plot(q4X, q4Y, 'DisplayName', 'Observed data');
hold on;
plot(q4X, q4polyfn(q4X), 'DisplayName', 'f(x) (ground truth)');

q4XMat = [];
q4rmse = [];
q4rmseOracle = [];

q5logLikelihood = [];
q5Aic = [];
q5Bic = [];

q6rmseTr = [];
q6rmseTe = [];
for curPow = 0:7
    q4XMat = [q4XMat reshape(q4X.^curPow, 11, 1)];
    q4w = (q4XMat.' * q4XMat)^(-1) * q4XMat.' * reshape(q4Y, 11, 1);
    q4YCur = reshape(q4XMat * q4w, 1, 11);
    plot(q4X, q4YCur, 'DisplayName', strcat('MLE for power ', num2str(curPow)));
    q4rmse = [q4rmse sqrt(sum((reshape(q4YCur, 1, 11) - q4Y).^2 / 11, 2))];
    q4rmseOracle = [q4rmseOracle sqrt(sum((reshape(q4YCur, 1, 11) - q4YOracle).^2 / 11, 2))];
    
    %q5
    q5var = sum((q4Y - q4YCur).^2, 2) / 11.0
    q5logLikelihoodCur = 11 * log(1/sqrt(2 * pi * q5var)) - 1/(2*q5var) * sum((q4Y - reshape(q4XMat * q4w, 1, 11)).^2, 2);
    q5logLikelihood = [q5logLikelihood q5logLikelihoodCur];
    q5Aic = [q5Aic (q5logLikelihoodCur - curPow + 1)];
    q5Bic = [q5Bic (q5logLikelihoodCur - 1 / 2 * (curPow + 1) * log(11))];
    
    %q6
    q6rmseTrCur = 0.0
    q6rmseTeCur = 0.0
    for j = 1:11
        q6XMatLoo = q4XMat;
        q6XExcLoo = q4XMat(j, :);
        q6XMatLoo(j, :) = [];
        
        q6YIncLoo = q4Y;
        q6YIncLoo(:, j) = [];
        q6YExcLoo = q4Y(:, j);
        
        q6w = (q6XMatLoo.' * q6XMatLoo)^(-1) * q6XMatLoo.' * reshape(q6YIncLoo, 10, 1);
        q6YIncLooPred = reshape(q6XMatLoo * q6w, 1, 10);
        q6YExcLooPred = q6XExcLoo * q6w;
        q6rmseTrCur = q6rmseTrCur + sqrt(sum((reshape(q6YIncLooPred, 1, 10) - q6YIncLoo).^2 / 10, 2));
        q6rmseTeCur = q6rmseTeCur + abs(q6YExcLooPred - q6YExcLoo);
    end
    q6rmseTr = [q6rmseTr (q6rmseTrCur / 11.0)];
    q6rmseTrCur / 11.0
    q6rmseTe = [q6rmseTe (q6rmseTeCur / 11.0)];
    q6rmseTeCur / 11.0
    
end

legend;
figure(9);
plot(0:7, q4rmse, 'DisplayName', 'RMS Error on observed data');
hold on;
plot(0:7, q4rmseOracle, 'DisplayName', 'RMS Error on oracle data');
legend;

%q5
figure(11);
plot(0:7, q5logLikelihood, 'DisplayName', 'Log Likelihood');
hold on;
plot(0:7, q5Aic, 'DisplayName', 'AIC');
plot(0:7, q5Bic, 'DisplayName', 'BIC');
legend;

%q6
figure(14);
plot(0:7, q6rmseTr, 'DisplayName', 'Error on training data');
hold on;
plot(0:7, q6rmseTe, 'DisplayName', 'Error on test data');
legend;

%q7

% I am regretting prefixing my variables with the question number.

% All metrics will end up being 101x7
q7RmseOracle = zeros(101, 7);
q7RmseObserved = zeros(101, 7);
q7LogLikelihood = zeros(101, 7);
q7Aic = zeros(101, 7);
q7Bic = zeros(101, 7);
q7RmseTr = zeros(101, 7);
q7RmseTe = zeros(101, 7);

q7X = q4X;
q7YOracle = q4YOracle;

for i = 1:101
    q7YObserved = q4polysamples(q7X);
    q7XMat = [];
    for curPow = 0:7
        q7XMat = [q7XMat reshape(q7X.^curPow, 11, 1)];
        q7W = (q7XMat.' * q7XMat)^(-1) * q7XMat.' * reshape(q7YObserved, 11, 1);
        q7YPredicted = reshape(q7XMat * q7W, 1, 11);
        q7RmseObserved(i, curPow + 1) = sqrt(sum((reshape(q7YPredicted, 1, 11) - q7YObserved).^2 / 11, 2));
        q7RmseOracle(i, curPow + 1) = sqrt(sum((reshape(q7YPredicted, 1, 11) - q7YOracle).^2 / 11, 2));
        
        q7Var = sum((q7YObserved - q7YPredicted).^2, 2) / 11.0;
        q7LogLikelihood(i, curPow + 1) = 11 * log(1/sqrt(2 * pi * q7Var)) - 1/(2*q7Var) * sum((q7YObserved - reshape(q7XMat * q7W, 1, 11)).^2, 2);
        q7Aic(i, curPow + 1) = q7LogLikelihood(i, curPow + 1) - curPow + 1;
        q7Bic(i, curPow + 1) = q7LogLikelihood(i, curPow + 1) - 1 / 2 * (curPow + 1) * log(11);
        
        q7RmseTrCur = 0.0;
        q7RmseTeCur = 0.0;
        for j = 1:11
            q7XIncLoo = q7XMat;
            q7XIncLoo(j, :) = [];
            q7XExcLoo = q7XMat(j, :);
            
            q7YObsIncLoo = q7YObserved;
            q7YObsIncLoo(:, j) = [];
            q7YObsExcLoo = q7YObserved(:, j);
            
            q7WFold = (q7XIncLoo.' * q7XIncLoo)^(-1) * q7XIncLoo.' * reshape(q7YObsIncLoo, 10, 1);
            q7YPredIncLoo = reshape(q7XIncLoo * q7WFold, 1, 10);
            q7YPredExcLoo = q7XExcLoo * q7WFold;
            q7RmseTrCur = q7RmseTrCur + sqrt(sum((reshape(q7YPredIncLoo, 1, 10) - q7YObsIncLoo).^2 / 10, 2));
            q7RmseTeCur = q7RmseTeCur + abs(q7YPredExcLoo - q7YObsExcLoo);
        end
        q7RmseTr(i, curPow + 1) = q7RmseTrCur / 11.0;
        q7RmseTe(i, curPow + 1) = q7RmseTeCur / 11.0;
        
    end
    figure(21);
    hold on;
    plot(0:7, q7Aic(i, :));
    
    figure(22);
    hold on;
    plot(0:7, q7Bic(i, :));
end

[q7RmseOracleMin, q7RmseOracleArgMin] = min(q7RmseOracle.');
q7RmseOracleArgMin = sort(q7RmseOracleArgMin.');
q7RmseOracleArgMin(51)

[q7RmseObservedMin, q7RmseObservedArgMin] = min(q7RmseObserved.');
q7RmseObservedArgMin = sort(q7RmseObservedArgMin.');
q7RmseObservedArgMin(51)

[q7LogLikelihoodMax, q7LogLikelihoodArgMax] = max(q7LogLikelihood.');
q7LogLikelihoodArgMax = sort(q7LogLikelihoodArgMax.');
q7LogLikelihoodArgMax(51)

[q7AicMax, q7AicArgMax] = max(q7Aic.');
q7AicArgMax = sort(q7AicArgMax.')
q7AicArgMax(51)

[q7BicMax, q7BicArgMax] = max(q7Bic.');
q7BicArgMax = sort(q7BicArgMax.')
q7BicArgMax(51)

[q7RmseTrMin, q7RmseTrArgMin] = min(q7RmseTr.');
q7RmseTrArgMin = sort(q7RmseTrArgMin.');
q7RmseTrArgMin(51)

[q7RmseTeMin, q7RmseTeArgMin] = min(q7RmseTe.');
q7RmseTeArgMin = sort(q7RmseTeArgMin.');
q7RmseTeArgMin(51)
