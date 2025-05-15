>> % Step 1: Read the three images
img1 = imread('W1.jpg'); 
img2 = imread('W2.jpg'); 
img3 = imread('W3.jpg');
>> % Step 3: Compute BRISQUE scores
score1 = brisque(img1);
score2 = brisque(img2);
score3 = brisque(img3);
>> % Step 4: Display scores
fprintf('BRISQUE score for Image 1: %.2f\n', score1);
fprintf('BRISQUE score for Image 2: %.2f\n', score2);
fprintf('BRISQUE score for Image 3: %.2f\n', score3);
BRISQUE score for Image 1: 31.10
BRISQUE score for Image 2: 37.48
BRISQUE score for Image 3: 54.73
>> % Step 5: Plot bar chart for comparison
scores = [score1, score2, score3];
images = {'Image 1', 'Image 2', 'Image 3'};

bar(scores);
set(gca, 'XTickLabel', images);
ylabel('BRISQUE Score');
title('BRISQUE Image Quality Comparison (Lower is Better)');
>> 
