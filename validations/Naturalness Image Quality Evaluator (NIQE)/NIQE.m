img1 = imread('W1.jpg');  % e.g., original or low quality 
img2 = imread('W3.jpg');  % e.g., denoised or upscaled version
>> % Step 2: Convert to grayscale if needed (NIQE works on grayscale images)
img1_gray = rgb2gray(W1); 
img2_gray = rgb2gray(W3);
Unrecognized function or variable 'W1'.
 
>> img1_gray = rgb2gray(img1);
img2_gray = rgb2gray(img2);
>> % Step 3: Compute NIQE scores
score1 = niqe(img1_gray);
score2 = niqe(img2_gray);
>> fprintf('NIQE score for Image 1: %.2f\n', score1);
fprintf('NIQE score for Image 2: %.2f\n', score2);
NIQE score for Image 1: 3.98
NIQE score for Image 2: 5.56
>> figure;
subplot(1,3,1);
imshow(img1);
title(['Image 1 - NIQE: ', num2str(score1, '%.2f')]);

subplot(1,3,2);
imshow(img2);
title(['Image 2 - NIQE: ', num2str(score2, '%.2f')]);
>> title(['Image 2 - NIQE: ', num2str(score2, '%.2f')]);
>> figure;
subplot(1,3,1);
imshow(img1);
title(['Image 1 - NIQE: ', num2str(score1, '%.2f')]);

subplot(1,3,2);
imshow(img2);
title(['Image 2 - NIQE: ', num2str(score2, '%.2f')]);
>> % NIQE scores for Image 1 and Image 2
score1 = 3.98;
score2 = 5.56;

niqe_scores = [score1, score2];
image_labels = {'Image 1', 'Image 2'};

% Create bar graph
figure;
bar(niqe_scores, 'FaceColor', [0.2 0.6 0.8]);  % Custom color
title('NIQE Score Comparison');
ylabel('NIQE Score (Lower is Better)');
set(gca, 'XTickLabel', image_labels);
ylim([0, max(niqe_scores) + 1]);

% Display score values on top of the bars
text(1:length(niqe_scores), niqe_scores, ...
     num2str(niqe_scores','%.2f'), ...
     'vert','bottom','horiz','center', ...
     'FontSize',10);

grid on;
>> 
