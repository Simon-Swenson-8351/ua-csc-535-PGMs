tigerImage = load('tiger.txt');
figure(1);
imshow(tigerImage);
firstRandomImage = rand(RandStream.getGlobalStream(), [236, 364]);
figure(2);
imshow(firstRandomImage);
secondRandomImage = rand(RandStream.getGlobalStream(), [236, 364]);
figure(3);
imshow(secondRandomImage);