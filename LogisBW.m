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
%  หมายเหตุการแก้ไข (ครั้งที่ 2 — แก้ปัญหา Recall ต่ำจาก Class Imbalance):
%   ปัญหาที่พบ: Recall = 0.042 (จับติดเชื้อได้แค่ 81/1929) แม้ Precision
%   จะพอใช้ได้ (0.551) แสดงว่าโมเดลเอนเอียงไปทาย "ปกติ" เกือบหมด สาเหตุคือ
%   ข้อมูลไม่สมดุล (ปกติ ~83% vs ติดเชื้อ ~17%) เหมือนที่เจอใน Linear
%   Regression วิธีแก้ที่เพิ่มเข้ามา:
%     A) แบ่ง Validation set (10%) แยกจาก Train เพื่อหา threshold ที่ให้
%        F1 สูงสุด แทนการ fix ไว้ที่ 0.5
%     B) Oversampling คลาสส่วนน้อย (Parasitized) เฉพาะใน training set
%        เท่านั้น (ทำหลัง split เพื่อไม่ให้เกิด data leakage)
%     C) พิมพ์สถิติของ probTest (min/max/mean) เพื่อวินิจฉัยการกระจายตัว
%
%  หมายเหตุการแก้ไข (ครั้งที่ 1 — รองรับภาพขาวดำ + ป้องกัน NaN):
%   - ฟังก์ชัน getCellFeatures ตรวจสอบอัตโนมัติว่าภาพเป็นสี (RGB) หรือขาวดำ
%     (grayscale) ถ้าเป็นภาพสี จะใช้ฟีเจอร์สี (mean R/G/B, colorVariance)
%     เหมือนเดิม แต่ถ้าเป็นภาพขาวดำ จะสลับไปใช้ฟีเจอร์พื้นผิวจาก GLCM
%     (Gray-Level Co-occurrence Matrix: contrast, homogeneity, energy,
%     correlation) แทน เพื่อไม่ให้ได้ฟีเจอร์ที่ซ้ำซ้อนกัน (R=G=B=meanGray)
%     ซึ่งจะไม่ช่วยแยกแยะคลาสเลย
%   - เพิ่มการป้องกัน NaN: graycoprops('Correlation') จะได้ NaN ถ้าภาพมี
%     ค่าสีเทาเดียวกันทั้งภาพ (เช่นภาพดำสนิท/ว่างเปล่า) เพราะต้องหารด้วย
%     std ที่เป็น 0 — ถ้าปล่อยผ่านไป จะทำให้ theta ทั้งก้อนกลายเป็น NaN
%     ตอนคูณเมทริกซ์ใน Gradient Descent (NaN แพร่กระจายไปทุกค่า) จึงต้อง
%     ดักแปลง NaN/Inf เป็น 0 ก่อนส่งฟีเจอร์ออกจากฟังก์ชัน
%   - จำนวนฟีเจอร์ยังคงเป็น 8 ตัวเท่าเดิมในทั้งสองกรณี เพื่อให้ส่วนอื่นของ
%     โค้ด (train/test, normalization, gradient descent) ทำงานได้โดยไม่ต้องแก้เพิ่ม
%   - ข้อควรระวัง: ควรใช้ภาพประเภทเดียวกันทั้งหมดในชุดข้อมูล (สีทั้งหมด
%     หรือขาวดำทั้งหมด) เพราะฟีเจอร์ทั้ง 8 ตัวจะมีความหมายต่างกันระหว่าง
%     สองโหมด ถ้าผสมกันโมเดลจะสับสน
%
%  โครงสร้างโฟลเดอร์ข้อมูล:
%   Logistic/
%     ├── Train Parasitized/   (รูปเซลล์ติดเชื้อสำหรับฝึกโมเดล)
%     └── Train Uninfected/    (รูปเซลล์ปกติสำหรับฝึกโมเดล)
%
%  วิธีใช้: แก้ path ตัวแปร datasetPath ให้ตรงกับเครื่องของคุณ แล้วรันสคริปต์
%  ข้อกำหนด: ต้องมี Image Processing Toolbox (ใช้ graycomatrix/graycoprops)
% =====================================================================

clear; clc; close all;

%% 1) กำหนดพาธข้อมูล
datasetPath    = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic';   % โฟลเดอร์หลักของ dataset
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

%% 3) ฟังก์ชันสกัดฟีเจอร์จากภาพ (Feature Extraction) — รองรับสีและขาวดำอัตโนมัติ
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

% --- ตรวจสอบภาพผิดปกติ (เช่นภาพดำสนิท/สีเดียวทั้งภาพ) ---
% ภาพเหล่านี้เคยทำให้ graycoprops คืนค่า NaN มาก่อน ตอนนี้ฟังก์ชัน
% getCellFeatures แปลง NaN เป็น 0 ให้แล้ว แต่ควรรู้จำนวนไว้เผื่อจำเป็น
% ต้องไปเช็คคุณภาพของไฟล์ภาพต้นฉบับเพิ่มเติม
suspiciousRows = find(X(:,1) == 0 & X(:,2) == 0);  % meanGray และ stdGray = 0
if ~isempty(suspiciousRows)
    fprintf('พบภาพที่น่าสงสัยว่าเป็นภาพดำสนิท/ว่างเปล่า: %d ภาพ (จากทั้งหมด %d ภาพ)\n', ...
        numel(suspiciousRows), size(X,1));
    fprintf('  → ฟีเจอร์ของภาพเหล่านี้ถูกตั้งเป็น 0 อัตโนมัติแล้ว แต่แนะนำให้ตรวจสอบ\n');
    fprintf('    ไฟล์ต้นฉบับว่าเป็นภาพเสียจริงหรือไม่ อาจพิจารณาคัดออกจาก dataset\n');
end

%% 5) แบ่งข้อมูล Train / Validation / Test (70% / 10% / 20%)
%    เพิ่ม Validation set เพื่อใช้หา threshold ที่ดีที่สุด โดยไม่แตะ test set
rng(42);
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

%% 6) ทำ Normalization ฟีเจอร์ (ยิ่งสำคัญกับ Gradient Descent เพราะช่วยให้ลู่เข้าเร็วขึ้น)
%    คำนวณ mu/sigma จาก training set (หลัง oversampling) เท่านั้น
muX  = mean(Xtrain, 1);
sigX = std(Xtrain, [], 1);
sigX(sigX == 0) = 1;

XtrainNorm = (Xtrain - muX) ./ sigX;
XvalNorm   = (Xval   - muX) ./ sigX;
XtestNorm  = (Xtest  - muX) ./ sigX;

XtrainAug = [ones(size(XtrainNorm,1),1), XtrainNorm];   % เพิ่ม bias term
XvalAug   = [ones(size(XvalNorm,1),1),   XvalNorm];
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

%% 9) ทำนายผลบน Train / Validation / Test Set
probTrain = sigmoid(XtrainAug * theta);   % ความน่าจะเป็นที่จะติดเชื้อ
probVal   = sigmoid(XvalAug   * theta);
probTest  = sigmoid(XtestAug  * theta);

% --- วินิจฉัย: ดูการกระจายของความน่าจะเป็นที่ทำนายได้ ---
fprintf('\n=== สถิติของความน่าจะเป็นที่ทำนาย (ก่อนตัด threshold) ===\n');
fprintf('probTrain : min=%.3f, max=%.3f, mean=%.3f\n', min(probTrain), max(probTrain), mean(probTrain));
fprintf('probVal   : min=%.3f, max=%.3f, mean=%.3f\n', min(probVal),   max(probVal),   mean(probVal));
fprintf('probTest  : min=%.3f, max=%.3f, mean=%.3f\n', min(probTest),  max(probTest),  mean(probTest));

%% 9.1) หา threshold ที่ดีที่สุดจาก Validation Set (ให้ F1 สูงสุด)
%      ไม่ใช้ test set ในการหา threshold เพื่อป้องกันผลประเมินที่ดีเกินจริง
threshRange = 0.05:0.01:0.95;
bestF1 = -1;
bestThreshold = 0.5;

for t = threshRange
    predVal = double(probVal >= t);
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
predTrain = double(probTrain >= threshold);
predTest  = double(probTest  >= threshold);

%% 10) ประเมินผลโมเดล
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
%     บันทึก isGrayModel ไว้ด้วย เพื่อให้สคริปต์ทำนายภายนอกรู้ว่าโมเดลนี้
%     เทรนด้วยฟีเจอร์แบบสีหรือแบบขาวดำ (GLCM) จะได้สกัดฟีเจอร์ให้ตรงกัน
firstImg = imread(fullfile(parasitizedFiles(1).folder, parasitizedFiles(1).name));
isGrayModel = (ndims(firstImg) == 2) || (size(firstImg, 3) == 1);

modelSavePath = fullfile(datasetPath, 'malaria_logistic_model.mat');
save(modelSavePath, 'theta', 'muX', 'sigX', 'imgSize', 'threshold', 'isGrayModel');
fprintf('\nบันทึกโมเดลเรียบร้อยที่: %s (isGrayModel = %d)\n', modelSavePath, isGrayModel);


%% =====================================================================
%  ฟังก์ชันสกัดฟีเจอร์จากภาพเซลล์ (รองรับภาพสีและภาพขาวดำอัตโนมัติ + ป้องกัน NaN)
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
    % ป้องกันไม่ให้ theta และ probTrain/probTest กลายเป็น NaN ทั้งหมดเมื่อ
    % มีภาพผิดปกติ (เช่นภาพเสีย/ภาพว่างเปล่า) ปนอยู่ในชุดข้อมูลแม้แต่ภาพเดียว
    feat(~isfinite(feat)) = 0;
end