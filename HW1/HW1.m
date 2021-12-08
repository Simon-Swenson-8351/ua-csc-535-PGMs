% Q1
randStream = RandStream('mrg32k3a', 'Seed', 0);
for i = 1:10
    throwResults = randi(randStream,6,[1, 1000]) + randi(randStream,6,[1, 1000]);
    counts = tabulate(throwResults);
    counts(12, 3)
end
randStream = RandStream('mrg32k3a', 'Seed', 0);
throwResults = randi(randStream,6,[1, 1000]) + randi(randStream,6,[1, 1000]);
counts = tabulate(throwResults);
counts(12, 3)

% Q2
tigerImage = load('tiger.txt');
figure(1);
imshow(tigerImage);
firstRandomImage = rand(RandStream.getGlobalStream(), [236, 364]);
figure(2);
imshow(firstRandomImage);
secondRandomImage = rand(RandStream.getGlobalStream(), [236, 364]);
figure(3);
imshow(secondRandomImage);

% Q3
X = linspace(-pi, pi);
figure(4);
plot(X, cos(X), 'r');
hold on;
plot(X, sin(X), 'b');
legend('cos(x)', 'sin(x)');

figure(10);
histogram(tigerImage);

figure(11);
histogram(tigerImage, 'Normalization', 'probability');

% Q4
mu = 70;
sigma = 2;
sampleStart = 68;
sampleEnd = 80;
sampleFreq = 2;
[x, y] = samples(sampleStart, sampleEnd, sampleFreq, mu, sigma);

figure(5);
x2 = linspace(60, 80);
size(x2)
size(normpdf(x2, mu, sigma))
plot(x2, normpdf(x2, mu, sigma));
hold on;
bar(x, y);

yArea = riemannSum(y, sampleFreq)

sampleStart = 20;
sampleEnd = 120;
target = 1;
sampleFreq = 10;
xSampleFreq = [];
yError = [];
while sampleFreq > 0.01
    xSampleFreq = [xSampleFreq, sampleFreq];
    [~, probValues] = samples(sampleStart, sampleEnd, sampleFreq, mu, sigma);
    yError = [yError, abs(target - riemannSum(probValues, sampleFreq))]; 
    sampleFreq = sampleFreq * 0.9;
end

figure(6);
plot(xSampleFreq, yError);

function[x, y] = samples(sampleStart, sampleEnd, sampleFreq, mu, sigma)
    x = linspace(sampleStart, sampleEnd, ((sampleEnd - sampleStart) / sampleFreq) + 1);
    y = normpdf(x, mu, sigma);
end

function[yArea] = riemannSum(y, width)
    yArea = sum(y * width);
end