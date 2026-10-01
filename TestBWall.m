%% =====================================================================
%  เปรียบเทียบ Accuracy ของ Linear Regression VS Logistic Regression
%  บนชุดภาพทดสอบทั้งโฟลเดอร์ (Batch Compare) - เวอร์ชันทดสอบด้วยภาพขาวดำ
% =====================================================================
clear; clc; close all;

datasetPath = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic';

%% 1) โหลดโมเดลทั้งสองตัว
linearModelFile   = fullfile(datasetPath, 'malaria_model.mat');
logisticModelFile = fullfile(datasetPath, 'malaria_logistic_model.mat');

if ~isfile(linearModelFile)
    error('ไม่พบไฟล์โมเดล Linear Regression: %s', linearModelFile);
end
if ~isfile(logisticModelFile)
    error('ไม่พบไฟล์โมเดล Logistic Regression: %s', logisticModelFile);
end

linModel = load(linearModelFile);
logModel = load(logisticModelFile);

imgSize  = linModel.imgSize;
sigmoid  = @(z) 1 ./ (1 + exp(-z));

%% 2) กำหนดโฟลเดอร์ทดสอบ (แก้ path ตรงนี้ถ้าจำเป็น)
testParasitizedDir = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic\BWTest Parasitized';
testUninfectedDir = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic\BWTest Uninfected';

testParaFiles = [dir(fullfile(testParasitizedDir, '*.png')); ...
                  dir(fullfile(testParasitizedDir, '*.jpg'))];
testUninFiles = [dir(fullfile(testUninfectedDir, '*.png')); ...
                  dir(fullfile(testUninfectedDir, '*.jpg'))];

allFiles   = [testParaFiles; testUninFiles];
trueLabels = [ones(numel(testParaFiles),1); zeros(numel(testUninFiles),1)];
numImages  = numel(allFiles);

fprintf('พบภาพทั้งหมด %d ภาพ (Parasitized=%d, Uninfected=%d)\n', ...
    numImages, numel(testParaFiles), numel(testUninFiles));

%% 3) ทำนายด้วยทั้งสองโมเดลทีละภาพ
predLin = zeros(numImages,1);
predLog = zeros(numImages,1);

fprintf('\nกำลังประมวลผล (โหมดภาพขาวดำ)...\n');
for i = 1:numImages
    imgPath = fullfile(allFiles(i).folder, allFiles(i).name);
    img = imread(imgPath);
    
    % สกัด Features โดยบังคับใช้โหมดขาวดำ
    feat = getCellFeatures(img, imgSize);
    
    % Linear
    featNormLin = (feat - linModel.muX) ./ linModel.sigX;
    yhatLin = [1, featNormLin] * linModel.theta;
    predLin(i) = double(yhatLin >= 0.5);
    
    % Logistic
    featNormLog = (feat - logModel.muX) ./ logModel.sigX;
    probLog = sigmoid([1, featNormLog] * logModel.theta);
    predLog(i) = double(probLog >= logModel.threshold);
    
    if mod(i, 100) == 0
        fprintf('  ประมวลผลแล้ว %d / %d ภาพ\n', i, numImages);
    end
end

%% 4) สรุปผลเปรียบเทียบ
accLin = mean(predLin == trueLabels) * 100;
accLog = mean(predLog == trueLabels) * 100;
agreeRate = mean(predLin == predLog) * 100;

fprintf('\n===================================================\n');
fprintf('                 สรุปผลเปรียบเทียบโมเดล\n');
fprintf('===================================================\n');
fprintf('จำนวนภาพทดสอบทั้งหมด      : %d ภาพ\n\n', numImages);
fprintf('Accuracy (Linear Regression)   : %.2f%%\n', accLin);
fprintf('Accuracy (Logistic Regression) : %.2f%%\n\n', accLog);
fprintf('อัตราที่สองโมเดลทำนาย "ตรงกัน" : %.2f%%\n', agreeRate);

% Confusion Matrix แยกแต่ละโมเดล
printConfusion('Linear Regression', predLin, trueLabels);
printConfusion('Logistic Regression', predLog, trueLabels);

%% 5) กราฟเปรียบเทียบ Accuracy
figure('Name', 'เปรียบเทียบความแม่นยำของสองโมเดล (โหมดขาวดำ)', 'NumberTitle', 'off');
bar([accLin, accLog]);
set(gca, 'XTickLabel', {'Linear Regression', 'Logistic Regression'});
ylabel('Accuracy (%)');
ylim([0 100]);
title('เปรียบเทียบความแม่นยำ: Linear vs Logistic Regression (Grayscale)');
grid on;

for i = 1:2
    vals = [accLin, accLog];
    text(i, vals(i)+2, sprintf('%.2f%%', vals(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

%% =====================================================================
%  ฟังก์ชันย่อย
%  =====================================================================
function printConfusion(modelName, pred, actual)
    TP = sum(pred == 1 & actual == 1);
    TN = sum(pred == 0 & actual == 0);
    FP = sum(pred == 1 & actual == 0);
    FN = sum(pred == 0 & actual == 1);
    
    fprintf('\n--- Confusion Matrix: %s ---\n', modelName);
    fprintf('                 ทำนาย: ติดเชื้อ   ทำนาย: ปกติ\n');
    fprintf('จริง: ติดเชื้อ      %5d              %5d\n', TP, FN);
    fprintf('จริง: ปกติ          %5d              %5d\n', FP, TN);
end

function feat = getCellFeatures(img, imgSize)
    % 1. แปลงภาพเป็น Grayscale ทันที
    if size(img,3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    
    % 2. ปรับขนาดภาพ
    grayImg = imresize(grayImg, imgSize);
    grayDouble = double(grayImg);
    
    % 3. คำนวณค่าทางสถิติพื้นฐานจากภาพขาวดำ
    meanGray = mean(grayDouble(:));
    stdGray = std(grayDouble(:));
    entropyVal = entropy(uint8(grayImg));
    
    edgeImg = edge(uint8(grayImg), 'Canny');
    edgeDensity = sum(edgeImg(:)) / numel(edgeImg);
    
    % 4. จำลองฟีเจอร์สี (RGB) เพื่อไม่ให้โมเดลเก่าพังจากจำนวนฟีเจอร์ไม่ตรงกัน
    % ถ้าคุณเทรนโมเดลใหม่ด้วยภาพขาวดำไปแล้ว และใช้แค่ 4-5 ฟีเจอร์ 
    % ให้แก้บรรทัด "feat = ..." ด้านล่างให้ตรงกับตอนเทรน
    meanR = meanGray;
    meanG = meanGray;
    meanB = meanGray;
    colorVariance = var([grayDouble(:); grayDouble(:); grayDouble(:)]);
    
    % คืนค่า 8 ฟีเจอร์โครงสร้างเดิม (แต่มาจากข้อมูลขาวดำทั้งหมด)
    feat = [meanR, meanG, meanB, meanGray, stdGray, entropyVal, edgeDensity, colorVariance];
end