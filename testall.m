%% =====================================================================
%  ทดสอบภาพทั้งโฟลเดอร์ (Batch Test) ด้วยโมเดล Linear Regression ที่ฝึกไว้แล้ว
%  =====================================================================
%  แนวคิด:
%   - วนอ่านภาพทั้งหมดในโฟลเดอร์ Test Parasitized และ Test Uninfected
%   - ทำนายผลทีละภาพด้วยโมเดลที่โหลดมาจาก malaria_model.mat
%   - สรุปผลรวม (accuracy, confusion matrix) และแสดงรายชื่อภาพที่ทายผิด
%
%  วิธีใช้: แก้ path testParasitizedDir / testUninfectedDir ให้ตรงกับเครื่องคุณ
%          แล้วรันสคริปต์ ไม่ต้องคลิกเลือกทีละภาพ
% =====================================================================

clear; clc; close all;

%% 1) โหลดโมเดลที่ฝึกไว้แล้ว
modelFile = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic\malaria_model.mat';

if ~isfile(modelFile)
    error('ไม่พบไฟล์โมเดล %s\nกรุณารันสคริปต์ฝึกโมเดล (Liner.m) ก่อน', modelFile);
end

loaded  = load(modelFile);
theta   = loaded.theta;
muX     = loaded.muX;
sigX    = loaded.sigX;
imgSize = loaded.imgSize;

%% 2) กำหนดโฟลเดอร์ที่ต้องการทดสอบทั้งหมด (แก้ path ตรงนี้ถ้าจำเป็น)
testParasitizedDir = 'D:\Malaria_Python_App\Photo\Parasitized';
testUninfectedDir  = 'D:\Malaria_Python_App\Photo\Uninfected';

testParaFiles = [dir(fullfile(testParasitizedDir, '*.png')); ...
                  dir(fullfile(testParasitizedDir, '*.jpg'))];
testUninFiles = [dir(fullfile(testUninfectedDir, '*.png')); ...
                  dir(fullfile(testUninfectedDir, '*.jpg'))];

fprintf('พบภาพใน Test Parasitized : %d ภาพ\n', numel(testParaFiles));
fprintf('พบภาพใน Test Uninfected  : %d ภาพ\n', numel(testUninFiles));

if isempty(testParaFiles) && isempty(testUninFiles)
    error('ไม่พบภาพในโฟลเดอร์ที่กำหนด กรุณาตรวจสอบ path');
end

%% 3) รวมรายการไฟล์ + label จริง (1 = ติดเชื้อ, 0 = ปกติ)
allFiles = [testParaFiles; testUninFiles];
trueLabels = [ones(numel(testParaFiles),1); zeros(numel(testUninFiles),1)];

numImages = numel(allFiles);
predLabels = zeros(numImages, 1);
rawScores  = zeros(numImages, 1);

%% 4) วนทำนายผลทีละภาพ
fprintf('\nกำลังประมวลผลภาพทั้งหมด %d ภาพ...\n', numImages);

threshold = 0.5;
for i = 1:numImages
    imgPath = fullfile(allFiles(i).folder, allFiles(i).name);
    img = imread(imgPath);

    feat = getCellFeatures(img, imgSize);
    featNorm = (feat - muX) ./ sigX;
    featAug  = [1, featNorm];

    yhat = featAug * theta;
    predLabels(i) = double(yhat >= threshold);
    rawScores(i)  = yhat;

    if mod(i, 50) == 0
        fprintf('  ประมวลผลแล้ว %d / %d ภาพ\n', i, numImages);
    end
end

%% 5) สรุปผลรวม
overallAcc = mean(predLabels == trueLabels) * 100;

TP = sum(predLabels == 1 & trueLabels == 1);
TN = sum(predLabels == 0 & trueLabels == 0);
FP = sum(predLabels == 1 & trueLabels == 0);
FN = sum(predLabels == 0 & trueLabels == 1);

precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
f1        = 2 * precision * recall / (precision + recall + eps);

fprintf('\n===================================================\n');
fprintf('           ผลสรุปการทดสอบทั้งโฟลเดอร์ (Batch Test)\n');
fprintf('===================================================\n');
fprintf('จำนวนภาพทั้งหมด    : %d ภาพ\n', numImages);
fprintf('Overall Accuracy   : %.2f%%\n\n', overallAcc);

fprintf('Confusion Matrix:\n');
fprintf('                 ทำนาย: ติดเชื้อ   ทำนาย: ปกติ\n');
fprintf('จริง: ติดเชื้อ      %5d              %5d\n', TP, FN);
fprintf('จริง: ปกติ          %5d              %5d\n', FP, TN);

fprintf('\nPrecision : %.3f\n', precision);
fprintf('Recall    : %.3f\n', recall);
fprintf('F1-score  : %.3f\n', f1);

%% 6) แยกดูผลเฉพาะกลุ่ม Test Parasitized (เพื่อเช็คปัญหาที่พบ)
paraIdx = trueLabels == 1;
paraAcc = mean(predLabels(paraIdx) == trueLabels(paraIdx)) * 100;
fprintf('\n--- เจาะดูเฉพาะ Test Parasitized ---\n');
fprintf('Accuracy เฉพาะกลุ่มติดเชื้อ : %.2f%% (%d จาก %d ภาพทายถูก)\n', ...
    paraAcc, TP, sum(paraIdx));

uninIdx = trueLabels == 0;
uninAcc = mean(predLabels(uninIdx) == trueLabels(uninIdx)) * 100;
fprintf('--- เจาะดูเฉพาะ Test Uninfected ---\n');
fprintf('Accuracy เฉพาะกลุ่มปกติ     : %.2f%% (%d จาก %d ภาพทายถูก)\n', ...
    uninAcc, TN, sum(uninIdx));

%% 7) แสดงตัวอย่างภาพที่ทายผิด (สูงสุด 12 ภาพ)
wrongIdx = find(predLabels ~= trueLabels);
fprintf('\nพบภาพที่ทายผิดทั้งหมด %d ภาพ จาก %d ภาพ\n', numel(wrongIdx), numImages);

if ~isempty(wrongIdx)
    numShow = min(12, numel(wrongIdx));
    figure('Name', 'ตัวอย่างภาพที่โมเดลทายผิด', 'NumberTitle', 'off');
    for k = 1:numShow
        idx = wrongIdx(k);
        imgPath = fullfile(allFiles(idx).folder, allFiles(idx).name);
        img = imread(imgPath);

        subplot(3,4,k);
        imshow(img);
        actualStr = {'ปกติ','ติดเชื้อ'};
        title(sprintf('จริง:%s\nทาย:%s (%.2f)', ...
            actualStr{trueLabels(idx)+1}, actualStr{predLabels(idx)+1}, rawScores(idx)), ...
            'FontSize', 8, 'Color', 'r');
    end
    sgtitle(sprintf('ตัวอย่างภาพที่ทายผิด (%d จาก %d ภาพ)', numel(wrongIdx), numImages));
end

%% 8) บันทึกผลลัพธ์ละเอียดเป็นตาราง (สำหรับเปิดดูใน Excel ได้)
fileNames = {allFiles.name}';
resultTable = table(fileNames, trueLabels, predLabels, rawScores, ...
    'VariableNames', {'FileName', 'ActualLabel', 'PredictedLabel', 'RawScore'});

resultCsvPath = fullfile(fileparts(modelFile), 'batch_test_results.csv');
writetable(resultTable, resultCsvPath);
fprintf('\nบันทึกผลลัพธ์ละเอียดไว้ที่: %s\n', resultCsvPath);


%% =====================================================================
%  ฟังก์ชันสกัดฟีเจอร์จากภาพเซลล์ (ต้องเหมือนกับตอนฝึกโมเดลทุกประการ)
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