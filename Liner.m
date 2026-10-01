%% =====================================================================
%  ตรวจจับเซลล์ติดเชื้อมาลาเรียด้วยโมเดล Linear Regression (ใช้เป็น Classifier)
%  =====================================================================
%  แนวคิด:
%   - Linear Regression ปกติใช้ทำนายค่าต่อเนื่อง แต่เราสามารถนำมาประยุกต์ใช้
%     เป็นตัวจำแนกประเภท (binary classification) ได้ โดยกำหนด label
%     0 = เซลล์ปกติ (Uninfected), 1 = เซลล์ติดเชื้อ (Parasitized)
%     แล้วใช้ผลทำนาย (y_hat) มาเทียบกับ threshold (เช่น 0.5) เพื่อตัดสินใจ
%
%  โครงสร้างโฟลเดอร์ข้อมูลที่ต้องเตรียม (ตัวอย่างจาก NIH Malaria Cell Dataset):
%   dataset/
%     ├── Parasitized/   (รูปเซลล์ติดเชื้อ, .png)
%     └── Uninfected/    (รูปเซลล์ปกติ, .png)
%
%  วิธีใช้: แก้ path ตัวแปร datasetPath ให้ตรงกับเครื่องของคุณ แล้วรันสคริปต์
% =====================================================================

clear; clc; close all;

%% 1) กำหนดพาธข้อมูล
datasetPath = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic';          % โฟลเดอร์หลักของ dataset
parasitizedDir = fullfile(datasetPath, 'Train Parasitized');
uninfectedDir  = fullfile(datasetPath, 'Train Uninfected');

imgSize = [50 50];   % ขนาดที่ใช้ resize ภาพก่อนสกัดฟีเจอร์

%% 2) โหลดรายชื่อไฟล์ภาพ
parasitizedFiles = [dir(fullfile(parasitizedDir, '*.png')); ...
                     dir(fullfile(parasitizedDir, '*.jpg'))];
uninfectedFiles  = [dir(fullfile(uninfectedDir, '*.png')); ...
                     dir(fullfile(uninfectedDir, '*.jpg'))];

fprintf('พบภาพเซลล์ติดเชื้อ (Parasitized): %d ภาพ\n', numel(parasitizedFiles));
fprintf('พบภาพเซลล์ปกติ (Uninfected):     %d ภาพ\n', numel(uninfectedFiles));

if isempty(parasitizedFiles) || isempty(uninfectedFiles)
    error(['ไม่พบไฟล์ภาพในโฟลเดอร์ที่กำหนด กรุณาตรวจสอบ path: %s'], datasetPath);
end

%% 3) ฟังก์ชันสกัดฟีเจอร์จากภาพ (Feature Extraction)
%    ใช้ค่าสถิติสี + พื้นผิว (texture) อย่างง่าย เพื่อให้ Linear Regression ใช้งานได้
extractFeatures = @(img) getCellFeatures(img, imgSize);

%% 4) สร้างชุดข้อมูล X (features) และ y (labels)
numPara = numel(parasitizedFiles);
numUnin = numel(uninfectedFiles);
numFeatures = 8;   % ตามจำนวนฟีเจอร์ใน getCellFeatures

X = zeros(numPara + numUnin, numFeatures);
y = zeros(numPara + numUnin, 1);

fprintf('\nกำลังสกัดฟีเจอร์จากภาพเซลล์ติดเชื้อ...\n');
for i = 1:numPara
    img = imread(fullfile(parasitizedFiles(i).folder, parasitizedFiles(i).name));
    X(i, :) = extractFeatures(img);
    y(i) = 1;   % ติดเชื้อ
end

fprintf('กำลังสกัดฟีเจอร์จากภาพเซลล์ปกติ...\n');
for i = 1:numUnin
    img = imread(fullfile(uninfectedFiles(i).folder, uninfectedFiles(i).name));
    X(numPara + i, :) = extractFeatures(img);
    y(numPara + i) = 0;   % ปกติ
end

%% 5) แบ่งข้อมูล Train / Test (80% / 20%)
rng(42);   % กำหนด seed เพื่อผลลัพธ์ที่ทำซ้ำได้
N = size(X, 1);
idx = randperm(N);
trainRatio = 0.8;
numTrain = round(trainRatio * N);

trainIdx = idx(1:numTrain);
testIdx  = idx(numTrain+1:end);

Xtrain = X(trainIdx, :);
ytrain = y(trainIdx);
Xtest  = X(testIdx, :);
ytest  = y(testIdx);

%% 6) ทำ Normalization ฟีเจอร์ (สำคัญมากสำหรับ Linear Regression)
muX  = mean(Xtrain, 1);
sigX = std(Xtrain, [], 1);
sigX(sigX == 0) = 1;  % ป้องกันหารด้วยศูนย์

XtrainNorm = (Xtrain - muX) ./ sigX;
XtestNorm  = (Xtest  - muX) ./ sigX;

%% 7) ฝึกโมเดล Linear Regression
%    เพิ่มคอลัมน์ bias (ค่าคงที่) ให้กับ X
XtrainAug = [ones(size(XtrainNorm,1),1), XtrainNorm];
XtestAug  = [ones(size(XtestNorm,1),1),  XtestNorm];

% หาค่าพารามิเตอร์ theta ด้วย Normal Equation: theta = (X'X)^-1 X'y
theta = pinv(XtrainAug' * XtrainAug) * XtrainAug' * ytrain;

fprintf('\n=== ค่าพารามิเตอร์ (theta) ที่ได้จากการฝึกโมเดล ===\n');
disp(theta');

%% 8) ทำนายผลบน Test Set
yhat_train = XtrainAug * theta;
yhat_test  = XtestAug  * theta;

threshold = 0.5;
predTrain = double(yhat_train >= threshold);
predTest  = double(yhat_test  >= threshold);

%% 9) ประเมินผลโมเดล
trainAcc = mean(predTrain == ytrain) * 100;
testAcc  = mean(predTest  == ytest ) * 100;

fprintf('\n=== ผลการประเมินโมเดล ===\n');
fprintf('ความแม่นยำบน Training set : %.2f%%\n', trainAcc);
fprintf('ความแม่นยำบน Test set     : %.2f%%\n', testAcc);

% Confusion Matrix
TP = sum(predTest == 1 & ytest == 1);
TN = sum(predTest == 0 & ytest == 0);
FP = sum(predTest == 1 & ytest == 0);
FN = sum(predTest == 0 & ytest == 1);

fprintf('\nConfusion Matrix (Test set):\n');
fprintf('                 ทำนาย: ติดเชื้อ   ทำนาย: ปกติ\n');
fprintf('จริง: ติดเชื้อ      %5d              %5d\n', TP, FN);
fprintf('จริง: ปกติ          %5d              %5d\n', FP, TN);

precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
f1        = 2 * precision * recall / (precision + recall + eps);

fprintf('\nPrecision : %.3f\n', precision);
fprintf('Recall    : %.3f\n', recall);
fprintf('F1-score  : %.3f\n', f1);

%% 10) แสดงตัวอย่างผลการทำนาย
figure('Name', 'ผลการตรวจจับเซลล์มาลาเรีย', 'NumberTitle', 'off');
numShow = min(9, numel(testIdx));
for k = 1:numShow
    sampleIdx = testIdx(k);
    if sampleIdx <= numPara
        img = imread(fullfile(parasitizedFiles(sampleIdx).folder, parasitizedFiles(sampleIdx).name));
    else
        localIdx = sampleIdx - numPara;
        img = imread(fullfile(uninfectedFiles(localIdx).folder, uninfectedFiles(localIdx).name));
    end
    subplot(3,3,k);
    imshow(img);
    actual = ytest(k);
    predicted = predTest(k);
    labelStr = {'ปกติ','ติดเชื้อ'};
    titleColor = 'k';
    if actual ~= predicted
        titleColor = 'r';
    end
    title(sprintf('จริง: %s | ทำนาย: %s', labelStr{actual+1}, labelStr{predicted+1}), ...
        'Color', titleColor, 'FontSize', 9);
end
sgtitle('ตัวอย่างผลการตรวจจับเซลล์มาลาเรียด้วย Linear Regression');

%% 11) บันทึกโมเดลไว้ใช้ทำนายภาพจากภายนอกในภายหลัง
%     ไฟล์นี้จะถูกโหลดโดยสคริปต์ test_external_image.m
modelSavePath = fullfile(datasetPath, 'malaria_model.mat');
save(modelSavePath, 'theta', 'muX', 'sigX', 'imgSize');
fprintf('\nบันทึกโมเดลเรียบร้อยที่: %s\n', modelSavePath);

%% =====================================================================
%  ฟังก์ชันสกัดฟีเจอร์จากภาพเซลล์
%  =====================================================================
function feat = getCellFeatures(img, imgSize)
    % แปลงเป็น RGB ถ้าเป็นภาพขาวดำ
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    img = imresize(img, imgSize);

    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));
    gray = double(rgb2gray(img));

    % ฟีเจอร์สี (มาลาเรียมักทำให้เม็ดเลือดมีจุดสีม่วง/น้ำเงินเข้มจากสี stain)
    meanR = mean(R(:));
    meanG = mean(G(:));
    meanB = mean(B(:));
    stdGray = std(gray(:));

    % ฟีเจอร์ความเข้ม (intensity) และ contrast
    meanGray = mean(gray(:));
    entropyVal = entropy(uint8(gray));

    % ฟีเจอร์ขอบ (edge density) บ่งบอกความไม่สม่ำเสมอของเซลล์ที่ติดเชื้อ
    edgeImg = edge(uint8(gray), 'Canny');
    edgeDensity = sum(edgeImg(:)) / numel(edgeImg);

    % ฟีเจอร์ความแปรปรวนของสี (บริเวณที่มีปรสิตมักมีสีเข้มไม่สม่ำเสมอ)
    colorVariance = var([R(:); G(:); B(:)]);

    feat = [meanR, meanG, meanB, meanGray, stdGray, entropyVal, edgeDensity, colorVariance];
end