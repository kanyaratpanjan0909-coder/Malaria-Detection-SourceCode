%% =====================================================================
%  ตรวจจับเซลล์ติดเชื้อมาลาเรียด้วยโมเดล Linear Regression (ใช้เป็น Classifier)
%  =====================================================================
%  แนวคิด:
%   - Linear Regression ปกติใช้ทำนายค่าต่อเนื่อง แต่เราสามารถนำมาประยุกต์ใช้
%     เป็นตัวจำแนกประเภท (binary classification) ได้ โดยกำหนด label
%     0 = เซลล์ปกติ (Uninfected), 1 = เซลล์ติดเชื้อ (Parasitized)
%     แล้วใช้ผลทำนาย (y_hat) มาเทียบกับ threshold เพื่อตัดสินใจ
%
%  หมายเหตุการแก้ไข (ครั้งที่ 2 — แก้ปัญหาโมเดลทำนายคลาสเดียวทั้งหมด):
%   ปัญหาที่พบ: TP=0 ทุกครั้ง (ไม่เคยทำนาย "ติดเชื้อ" เลย) สาเหตุคือ
%     1) ข้อมูลไม่สมดุล (Uninfected เยอะกว่า Parasitized มาก ~83%/17%)
%        ทำให้ Normal Equation ดึงค่าทำนาย (yhat) ให้ต่ำกว่า 0.5 เกือบหมด
%     2) threshold 0.5 ตายตัว ไม่เหมาะกับข้อมูลไม่สมดุลแบบนี้
%   วิธีแก้ที่เพิ่มเข้ามา:
%     A) พิมพ์สถิติของ yhat_test (min/max/mean) เพื่อวินิจฉัยปัญหา
%     B) ทำ Oversampling คลาสส่วนน้อย (Parasitized) เฉพาะใน "training set"
%        เท่านั้น (ทำหลัง split แล้ว เพื่อไม่ให้ข้อมูลเดียวกันไปอยู่ทั้ง
%        train และ test/val — ป้องกัน data leakage)
%     C) แบ่ง validation set ออกจาก training set เพิ่ม เพื่อใช้หา
%        threshold ที่ดีที่สุด (ให้ F1 สูงสุด) แล้วค่อยนำ threshold นั้น
%        ไปใช้ประเมินผลจริงบน test set (ไม่ tune threshold จาก test set
%        โดยตรง เพราะจะทำให้ผลลัพธ์ดูดีเกินจริง/overfit กับ test)
%
%  หมายเหตุการแก้ไข (ครั้งที่ 1 — รองรับภาพขาวดำ):
%   - ฟังก์ชัน getCellFeatures ตรวจสอบอัตโนมัติว่าภาพเป็นสี (RGB) หรือขาวดำ
%     (grayscale) ถ้าเป็นภาพสี จะใช้ฟีเจอร์สี (mean R/G/B, colorVariance)
%     เหมือนเดิม แต่ถ้าเป็นภาพขาวดำ จะสลับไปใช้ฟีเจอร์พื้นผิวจาก GLCM
%     (Contrast, Homogeneity, Energy, Correlation) แทน
%   - จำนวนฟีเจอร์ยังคงเป็น 8 ตัวเท่าเดิมในทั้งสองกรณี
%
%  โครงสร้างโฟลเดอร์ข้อมูลที่ต้องเตรียม:
%   dataset/
%     ├── BWTrain Parasitized/   (รูปเซลล์ติดเชื้อ, .png)
%     └── BWTrain Uninfected/    (รูปเซลล์ปกติ, .png)
%
%  วิธีใช้: แก้ path ตัวแปร datasetPath ให้ตรงกับเครื่องของคุณ แล้วรันสคริปต์
%  ข้อกำหนด: ต้องมี Image Processing Toolbox (ใช้ graycomatrix/graycoprops)
% =====================================================================

clear; clc; close all;

%% 1) กำหนดพาธข้อมูล
datasetPath = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic';          % โฟลเดอร์หลักของ dataset
parasitizedDir = fullfile(datasetPath, 'BWTrain Parasitized');
uninfectedDir  = fullfile(datasetPath, 'BWTrain Uninfected');

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
%    (รองรับทั้งภาพสีและภาพขาวดำโดยอัตโนมัติ ดูรายละเอียดในฟังก์ชันด้านล่าง)
extractFeatures = @(img) getCellFeatures(img, imgSize);

%% 4) สร้างชุดข้อมูล X (features) และ y (labels)
numPara = numel(parasitizedFiles);
numUnin = numel(uninfectedFiles);
numFeatures = 8;   % ตามจำนวนฟีเจอร์ใน getCellFeatures (คงที่ทั้งสองโหมด)

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

fprintf('\nสัดส่วนคลาสทั้งหมด: ติดเชื้อ %d (%.1f%%) | ปกติ %d (%.1f%%)\n', ...
    numPara, 100*numPara/(numPara+numUnin), numUnin, 100*numUnin/(numPara+numUnin));

%% 5) แบ่งข้อมูล Train / Validation / Test (70% / 10% / 20%)
%    เพิ่ม Validation set เพื่อใช้หา threshold ที่ดีที่สุด โดยไม่แตะ test set
rng(42);   % กำหนด seed เพื่อผลลัพธ์ที่ทำซ้ำได้
N = size(X, 1);
idx = randperm(N);

trainRatio = 0.7;
valRatio   = 0.1;
numTrain = round(trainRatio * N);
numVal   = round(valRatio   * N);

trainIdx = idx(1:numTrain);
valIdx   = idx(numTrain+1 : numTrain+numVal);
testIdx  = idx(numTrain+numVal+1 : end);

Xtrain = X(trainIdx, :);  ytrain = y(trainIdx);
Xval   = X(valIdx, :);    yval   = y(valIdx);
Xtest  = X(testIdx, :);   ytest  = y(testIdx);

fprintf('\nจำนวนข้อมูล: Train = %d | Validation = %d | Test = %d\n', ...
    numel(ytrain), numel(yval), numel(ytest));

%% 5.1) Oversampling คลาสส่วนน้อยใน Training Set เท่านั้น
%     ทำหลัง split แล้ว เพื่อไม่ให้ตัวอย่างที่ถูกสุ่มซ้ำไปปนอยู่ใน
%     validation/test set (ป้องกัน data leakage ซึ่งจะทำให้ผลประเมินเพี้ยน)
classCounts = [sum(ytrain==0), sum(ytrain==1)];
fprintf('ก่อน oversampling (Train): ปกติ = %d, ติดเชื้อ = %d\n', classCounts(1), classCounts(2));

minorityLabel = 0; if classCounts(1) > classCounts(2), minorityLabel = 1; end
majorityCount = max(classCounts);
minorityIdxInTrain = find(ytrain == minorityLabel);
numToAdd = majorityCount - numel(minorityIdxInTrain);

if numToAdd > 0
    resampleIdx = minorityIdxInTrain(randi(numel(minorityIdxInTrain), numToAdd, 1));
    Xtrain = [Xtrain; Xtrain(resampleIdx, :)];
    ytrain = [ytrain; ytrain(resampleIdx)];
end

fprintf('หลัง oversampling (Train):  ปกติ = %d, ติดเชื้อ = %d\n', ...
    sum(ytrain==0), sum(ytrain==1));

%% 6) ทำ Normalization ฟีเจอร์ (สำคัญมากสำหรับ Linear Regression)
%    คำนวณ mu/sigma จาก training set (หลัง oversampling) เท่านั้น
muX  = mean(Xtrain, 1);
sigX = std(Xtrain, [], 1);
sigX(sigX == 0) = 1;  % ป้องกันหารด้วยศูนย์

XtrainNorm = (Xtrain - muX) ./ sigX;
XvalNorm   = (Xval   - muX) ./ sigX;
XtestNorm  = (Xtest  - muX) ./ sigX;

%% 7) ฝึกโมเดล Linear Regression
%    เพิ่มคอลัมน์ bias (ค่าคงที่) ให้กับ X
XtrainAug = [ones(size(XtrainNorm,1),1), XtrainNorm];
XvalAug   = [ones(size(XvalNorm,1),1),   XvalNorm];
XtestAug  = [ones(size(XtestNorm,1),1),  XtestNorm];

% หาค่าพารามิเตอร์ theta ด้วย Normal Equation: theta = (X'X)^-1 X'y
theta = pinv(XtrainAug' * XtrainAug) * XtrainAug' * ytrain;

fprintf('\n=== ค่าพารามิเตอร์ (theta) ที่ได้จากการฝึกโมเดล ===\n');
disp(theta');

%% 8) ทำนายค่า yhat บนทั้งสามชุด
yhat_train = XtrainAug * theta;
yhat_val   = XvalAug   * theta;
yhat_test  = XtestAug  * theta;

% --- วินิจฉัย: ดูการกระจายของ yhat เพื่อเช็คว่าติดปัญหาเดิมอีกหรือไม่ ---
fprintf('\n=== สถิติของค่าทำนาย yhat (ก่อนตัด threshold) ===\n');
fprintf('yhat_train : min=%.3f, max=%.3f, mean=%.3f\n', min(yhat_train), max(yhat_train), mean(yhat_train));
fprintf('yhat_val   : min=%.3f, max=%.3f, mean=%.3f\n', min(yhat_val),   max(yhat_val),   mean(yhat_val));
fprintf('yhat_test  : min=%.3f, max=%.3f, mean=%.3f\n', min(yhat_test),  max(yhat_test),  mean(yhat_test));

%% 8.1) หา threshold ที่ดีที่สุดจาก Validation Set (ให้ F1 สูงสุด)
%      ไม่ใช้ test set ในการหา threshold เพื่อป้องกันผลประเมินที่ดีเกินจริง
threshRange = 0.05:0.01:0.95;
bestF1 = -1;
bestThreshold = 0.5;

for t = threshRange
    predVal = double(yhat_val >= t);
    tp = sum(predVal == 1 & yval == 1);
    fp = sum(predVal == 1 & yval == 0);
    fn = sum(predVal == 0 & yval == 1);
    p = tp / (tp + fp + eps);
    r = tp / (tp + fn + eps);
    f1_t = 2 * p * r / (p + r + eps);
    if f1_t > bestF1
        bestF1 = f1_t;
        bestThreshold = t;
    end
end

fprintf('\nThreshold ที่ดีที่สุดจาก Validation Set = %.2f (F1 บน validation = %.3f)\n', ...
    bestThreshold, bestF1);

threshold = bestThreshold;   % ใช้ threshold ที่ได้จาก validation แทนค่า fix 0.5
predTrain = double(yhat_train >= threshold);
predTest  = double(yhat_test  >= threshold);

%% 9) ประเมินผลโมเดล
trainAcc = mean(predTrain == ytrain) * 100;
testAcc  = mean(predTest  == ytest ) * 100;

fprintf('\n=== ผลการประเมินโมเดล (threshold = %.2f) ===\n', threshold);
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
%     หมายเหตุ: หลัง oversampling ดัชนี trainIdx เดิมใช้ไม่ได้กับ Xtrain
%     ที่ถูกต่อแถวเพิ่มแล้ว แต่ testIdx ยังคงอ้างอิงไฟล์ต้นฉบับได้ตรง
%     เพราะ test set ไม่ถูก oversample จึงใช้แสดงตัวอย่างได้ตามปกติ
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
%     บันทึก threshold ที่ปรับให้เหมาะกับข้อมูลไม่สมดุลไว้ด้วย แทนที่จะ
%     ใช้ 0.5 ตายตัว มิฉะนั้นสคริปต์ทำนายภายนอกจะเจอปัญหาเดิม
firstImg = imread(fullfile(parasitizedFiles(1).folder, parasitizedFiles(1).name));
isGrayModel = (ndims(firstImg) == 2) || (size(firstImg, 3) == 1);

modelSavePath = fullfile(datasetPath, 'malaria_model.mat');
save(modelSavePath, 'theta', 'muX', 'sigX', 'imgSize', 'isGrayModel', 'threshold');
fprintf('\nบันทึกโมเดลเรียบร้อยที่: %s (isGrayModel = %d, threshold = %.2f)\n', ...
    modelSavePath, isGrayModel, threshold);

%% =====================================================================
%  ฟังก์ชันสกัดฟีเจอร์จากภาพเซลล์ (รองรับภาพสีและภาพขาวดำอัตโนมัติ)
%  =====================================================================
function feat = getCellFeatures(img, imgSize)
    % ตรวจสอบว่าภาพต้นฉบับเป็นภาพขาวดำ (2-D, ไม่มีมิติสี) หรือภาพสี (3-D)
    % ต้องเช็คก่อน resize เพราะ imresize ไม่เปลี่ยนจำนวนมิติ แต่เช็คตรงนี้
    % ชัดเจนกว่าและตรงกับข้อมูลจริงของไฟล์ที่ imread เข้ามา
    isGrayInput = (ndims(img) == 2) || (size(img, 3) == 1);

    if isGrayInput
        img = cat(3, img, img, img);   % ทำให้ imresize/แสดงผลทำงานได้ปกติ
    end
    img = imresize(img, imgSize);
    gray = double(rgb2gray(img));

    % ฟีเจอร์พื้นฐานที่ใช้ได้ทั้งสองโหมด
    meanGray   = mean(gray(:));
    stdGray    = std(gray(:));
    entropyVal = entropy(uint8(gray));
    edgeImg      = edge(uint8(gray), 'Canny');
    edgeDensity  = sum(edgeImg(:)) / numel(edgeImg);

    if isGrayInput
        % --- โหมดภาพขาวดำ ---
        % ไม่มีข้อมูลสี (R=G=B) จึงไม่ใช้ meanR/meanG/meanB/colorVariance
        % เพราะจะซ้ำซ้อนกับ meanGray/stdGray โดยสมบูรณ์ (ให้ข้อมูลเพิ่ม = 0)
        % แทนที่ด้วยฟีเจอร์พื้นผิวจาก GLCM (Gray-Level Co-occurrence Matrix)
        % ซึ่งบ่งบอกความไม่สม่ำเสมอของเนื้อเซลล์ได้ดีกว่าในกรณีไม่มีสี
        glcm = graycomatrix(uint8(gray), 'Offset', [0 1; -1 1; -1 0; -1 -1], ...
                             'Symmetric', true);
        stats = graycoprops(glcm, {'Contrast', 'Homogeneity', 'Energy', 'Correlation'});

        % graycoprops('Correlation') จะได้ NaN ถ้าภาพมีค่าสีเทาเดียวกัน
        % ทั้งภาพ (เช่นภาพดำสนิท) เพราะต้องหารด้วย std ที่เป็น 0
        % แก้โดยแทนค่า NaN ด้วย 0 (แปลว่า "ไม่มีความสัมพันธ์เชิงพื้นผิว")
        glcmContrast     = mean(stats.Contrast);
        glcmHomogeneity  = mean(stats.Homogeneity);
        glcmEnergy       = mean(stats.Energy);
        glcmCorrelation  = mean(stats.Correlation);
        if isnan(glcmCorrelation), glcmCorrelation = 0; end
        if isnan(glcmContrast),    glcmContrast    = 0; end
        if isnan(glcmHomogeneity), glcmHomogeneity = 0; end
        if isnan(glcmEnergy),      glcmEnergy      = 0; end

        feat = [meanGray, stdGray, entropyVal, edgeDensity, ...
                glcmContrast, glcmHomogeneity, glcmEnergy, glcmCorrelation];
    else
        % --- โหมดภาพสี (เดิม) ---
        R = double(img(:,:,1));
        G = double(img(:,:,2));
        B = double(img(:,:,3));

        meanR = mean(R(:));
        meanG = mean(G(:));
        meanB = mean(B(:));

        % ฟีเจอร์ความแปรปรวนของสี (บริเวณที่มีปรสิตมักมีสีเข้มไม่สม่ำเสมอ)
        colorVariance = var([R(:); G(:); B(:)]);

        feat = [meanR, meanG, meanB, meanGray, stdGray, entropyVal, ...
                edgeDensity, colorVariance];
    end

    % --- ตัวป้องกันสุดท้าย: แปลง NaN หรือ Inf ใด ๆ ที่หลุดรอดมาให้เป็น 0 ---
    % ป้องกันไม่ให้ theta และ yhat กลายเป็น NaN ทั้งหมดเมื่อมีภาพผิดปกติ
    % (เช่นภาพเสีย/ภาพว่างเปล่า) ปนอยู่ในชุดข้อมูลแม้แต่ภาพเดียว
    feat(~isfinite(feat)) = 0;
end