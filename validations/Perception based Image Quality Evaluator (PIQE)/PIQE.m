>> img1 = imread('W1.jpg'); 
img2 = imread('W2.jpg'); 
img3 = imread('W3.jpg');
>> % Step 2: Convert images to grayscale if they are RGB
if size(img1, 3) == 3
    img1 = rgb2gray(img1);
end
if size(img2, 3) == 3
    img2 = rgb2gray(img2);
end
if size(img3, 3) == 3
    img3 = rgb2gray(img3);
end
>> % Step 3: Compute PIQE Scores
score1 = piqe(img1);
score2 = piqe(img2);
score3 = piqe(img3);
>> Step 4: Display the Scores
fprintf('PIQE score for Image 1: %.2f\n', score1);
fprintf('PIQE score for Image 2: %.2f\n', score2);
fprintf('PIQE score for Image 3: %.2f\n', score3);
 
>> fprintf('PIQE score for Image 1: %.2f\n', score1);
fprintf('PIQE score for Image 2: %.2f\n', score2);
fprintf('PIQE score for Image 3: %.2f\n', score3);
PIQE score for Image 1: 9.93
PIQE score for Image 2: 17.17
PIQE score for Image 3: 68.93
>> scores = [score1, score2, score3];
imageLabels = {'Image 1', 'Image 2', 'Image 3'};
>> figure;
bar(scores);
set(gca, 'XTickLabel', imageLabels);
ylabel('PIQE Score');
title('Perception-based Image Quality (PIQE) Scores');
grid on;
>> 
