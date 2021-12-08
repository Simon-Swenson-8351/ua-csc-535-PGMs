% Tasks to do:
% Sample the distribution function every delta steps to estimate the
% cumulative distribution function. Store the result in a key -> value map
% or array (which we can combine with the delta to derive they key for a 
% given value).
% Reverse the key-value map to estimate F^(-1).
classdef DistributionSampleGenerator
    properties
        PDF
        SampleStart
        SampleStop
        Delta
        RandomNumberGenerator
        Samples
    end
    methods
        % Constructor
        function obj = DistributionSampleGenerator(pdf, sampleStart, sampleStop, delta, randomNumberGenerator)
            obj.PDF = pdf;
            obj.SampleStart = sampleStart;
            obj.SampleStop = sampleStop;
            obj.Delta = delta;
            obj.RandomNumberGenerator = randomNumberGenerator;
            obj.Samples = obj.buildSamples();
        end

        % Build our sample array for the CDF, from which we will derive our generator.
        function samples = buildSamples(obj)
            arySize = floor((obj.SampleStop - obj.SampleStart) / obj.Delta);
            samples = zeros(arySize, 1);
            for i = 1:arySize
                accum = obj.PDF(obj.SampleStart + (i - 1) * obj.Delta) * obj.Delta;
                if i > 1
                    accum = accum + samples(i - 1);
                end
                samples(i) = accum;
            end
        end

        % Random sampling function
        function result = sample(obj)
            val = obj.RandomNumberGenerator.rand();
            for i = 1:size(obj.Samples, 1)
                if obj.Samples(i) > val
                    result = obj.SampleStart + obj.Delta * (i - 2);
                    return;
                end
            end
            result = obj.SampleStart + obj.Delta * floor((obj.SampleStop - obj.SampleStart) / obj.Delta);
        end
    end
end