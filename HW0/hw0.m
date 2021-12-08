function [num_questions] = hw0(infile)
% Reads an image, then asks a series of questions
image = imread(infile);
figure(1);
imshow(image);
[num_rows, num_cols, num_channels] = size(image);

red_min = min(min(image(:, :, 1)));
red_max = max(max(image(:, :, 1)));
green_min = min(min(image(:, :, 2)));
green_max = max(max(image(:, :, 2)));
blue_min = min(min(image(:, :, 2)));
blue_max = max(max(image(:, :, 2)));

red_channel = image(:, :, 1);
green_channel = image(:, :, 2);
blue_channel = image(:, :, 3);

image_crazy = zeros(size(image), 'uint8');
image_crazy(:, :, 1) = green_channel;
image_crazy(:, :, 2) = blue_channel;
image_crazy(:, :, 3) = red_channel;
figure(2);
imshow(image_crazy);

image_scaled = double(rgb2gray(image)) / 255.0;
image_hw = size(image_scaled);
for i = 1:image_hw(1)
    for j = 1:image_hw(2)
        if mod(i, 5) == 0 && mod(j, 5) == 0
            image_scaled(i, j) = 1.0;
        end
    end
end

figure(3);
histogram(image_scaled(:));

figure(4);
X = linspace(-pi, pi);
plot(X, sin(X), 'r');
hold on;
plot(X, cos(X), 'g');