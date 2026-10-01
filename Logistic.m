%% =====================================================================
%  ตรวจจับเซลล์ติดเชื้อมาลาเรียด้วยโมเดล Logistic Regression
%  =====================================================================
%  แนวคิด:
%   - Logistic Regression เหมาะกับงาน Binary Classification มากกว่า
%     Linear Regression เพราะใช้ฟังก์ชัน Sigmoid บีบค่าให้อยู่ในช่วง [0,1]
%     ตีความได้ตรงตัวว่าเป็น "ความน่าจะเป็น" ที่เซลล์จะติดเชื้อ
%     label: 0 = เซลล์ปกติ (Uninfected), 1 = เซลล์ติดเชื้อ (Parasitized)
%
%  วิธีฝึกโมเดล: ใช้ Gradient Descent ปรับค่า theta เพื่อลด
%  Cross-Entropy Loss (Log Loss) แทนการใช้ Normal Equation แบบ Linear Regression
%
%  โครงสร้างโฟลเดอร์ข้อมูล:
%   Logistic/
%     ├── Train Parasitized/   (รูปเซลล์ติดเชื้อสำหรับฝึกโมเดล)
%     └── Train Uninfected/    (รูปเซลล์ปกติสำหรับฝึกโมเดล)
%
%  วิธีใช้: แก้ path ตัวแปร datasetPath ให้ตรงกับเครื่องของคุณ แล้วรันสคริปต์
% =====================================================================

clear; clc; close all;

%% 1) กำหนดพาธข้อมูล
datasetPath    = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic';   % โฟลเดอร์หลักของ dataset
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

%% 3) ฟังก์ชันสกัดฟีเจอร์จากภาพ (Feature Extraction) — เหมือนกับเวอร์ชัน Linear Regression
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
rng(42);
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

%% 6) ทำ Normalization ฟีเจอร์ (ยิ่งสำคัญกับ Gradient Descent เพราะช่วยให้ลู่เข้าเร็วขึ้น)
muX  = mean(Xtrain, 1);
sigX = std(Xtrain, [], 1);
sigX(sigX == 0) = 1;

XtrainNorm = (Xtrain - muX) ./ sigX;
XtestNorm  = (Xtest  - muX) ./ sigX;

XtrainAug = [ones(size(XtrainNorm,1),1), XtrainNorm];   % เพิ่ม bias term
XtestAug  = [ones(size(XtestNorm,1),1),  XtestNorm];

%% 7) ฝึกโมเดล Logistic Regression ด้วย Gradient Descent
numParams = size(XtrainAug, 2);
theta = zeros(numParams, 1);     % เริ่มต้นค่า theta ที่ศูนย์

alpha = 0.1;          % learning rate
numIters = 3000;      % จำนวนรอบการอัปเดต
mTrain = size(XtrainAug, 1);

costHistory = zeros(numIters, 1);

sigmoid = @(z) 1 ./ (1 + exp(-z));

fprintf('\nกำลังฝึกโมเดล Logistic Regression ด้วย Gradient Descent...\n');
for iter = 1:numIters
    z = XtrainAug * theta;
    h = sigmoid(z);

    % Gradient ของ Cross-Entropy Loss
    grad = (1/mTrain) * (XtrainAug' * (h - ytrain));
    theta = theta - alpha * grad;

    % คำนวณ Cost (Cross-Entropy / Log Loss) เพื่อดูการลู่เข้า
    epsVal = 1e-10;   % ป้องกัน log(0)
    cost = -(1/mTrain) * sum(ytrain .* log(h + epsVal) + (1 - ytrain) .* log(1 - h + epsVal));
    costHistory(iter) = cost;

    if mod(iter, 500) == 0
        fprintf('  รอบที่ %5d / %d | Cost = %.4f\n', iter, numIters, cost);
    end
end

fprintf('\n=== ค่าพารามิเตอร์ (theta) ที่ได้จากการฝึกโมเดล ===\n');
disp(theta');

%% 8) กราฟแสดงการลู่เข้าของ Cost (ตรวจสอบว่า Gradient Descent ทำงานถูกต้อง)
figure('Name', 'Cost Convergence', 'NumberTitle', 'off');
plot(1:numIters, costHistory, 'LineWidth', 1.5);
xlabel('รอบการฝึก (Iteration)');
ylabel('Cost (Cross-Entropy Loss)');
title('การลู่เข้าของ Cost ระหว่างฝึกโมเดล Logistic Regression');
grid on;

%% 9) ทำนายผลบน Train / Test Set
threshold = 0.5;

probTrain = sigmoid(XtrainAug * theta);   % ความน่าจะเป็นที่จะติดเชื้อ
probTest  = sigmoid(XtestAug  * theta);

predTrain = double(probTrain >= threshold);
predTest  = double(probTest  >= threshold);

%% 10) ประเมินผลโมเดล
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

%% 11) แสดงตัวอย่างผลการทำนาย พร้อมค่าความน่าจะเป็น
figure('Name', 'ผลการตรวจจับเซลล์มาลาเรีย (Logistic Regression)', 'NumberTitle', 'off');
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
    prob = probTest(k);
    labelStr = {'ปกติ','ติดเชื้อ'};
    titleColor = 'k';
    if actual ~= predicted
        titleColor = 'r';
    end
    title(sprintf('จริง: %s | ทาย: %s (%.2f)', ...
        labelStr{actual+1}, labelStr{predicted+1}, prob), ...
        'Color', titleColor, 'FontSize', 9);
end
sgtitle('ตัวอย่างผลการตรวจจับเซลล์มาลาเรียด้วย Logistic Regression');

%% 12) บันทึกโมเดลไว้ใช้ทำนายภาพจากภายนอกในภายหลัง
%     ตั้งชื่อไฟล์แยกจาก Linear Regression เพื่อไม่ให้ทับกัน
modelSavePath = fullfile(datasetPath, 'malaria_logistic_model.mat');
save(modelSavePath, 'theta', 'muX', 'sigX', 'imgSize', 'threshold');
fprintf('\nบันทึกโมเดลเรียบร้อยที่: %s\n', modelSavePath);


%% =====================================================================
%  ฟังก์ชันสกัดฟีเจอร์จากภาพเซลล์ (เหมือนกับเวอร์ชัน Linear Regression ทุกประการ)
%  =====================================================================
function feat = getCellFeatures(img, imgSize)
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    img = imresize(img, imgSize);

    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));
    gray = double(rgb2gray(img));

    meanR = mean(R(:));
    meanG = mean(G(:));
    meanB = mean(B(:));
    stdGray = std(gray(:));

    meanGray = mean(gray(:));
    entropyVal = entropy(uint8(gray));

    edgeImg = edge(uint8(gray), 'Canny');
    edgeDensity = sum(edgeImg(:)) / numel(edgeImg);

    colorVariance = var([R(:); G(:); B(:)]);

    feat = [meanR, meanG, meanB, meanGray, stdGray, entropyVal, edgeDensity, colorVariance];
end