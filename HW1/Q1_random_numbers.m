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