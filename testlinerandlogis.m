%% =====================================================================
%  เปรียบเทียบผลการตรวจจับเซลล์มาลาเรีย: Linear Regression VS Logistic Regression
%  =====================================================================
%  แนวคิด:
%   - โหลดโมเดลทั้ง 2 ตัว (Linear และ Logistic) ที่ฝึกไว้แล้ว
%   - เลือกภาพเดียวกัน แล้วให้ทั้ง 2 โมเดลทำนายพร้อมกัน
%   - แสดงผลเปรียบเทียบ ว่าตรงกันหรือไม่ ต่างกันอย่างไร
%
%  ข้อกำหนดก่อนรัน:
%   - ต้องรันสคริปต์ฝึกโมเดล Linear Regression (Liner.m) แล้ว
%     -> ได้ไฟล์ malaria_model.mat
%   - ต้องรันสคริปต์ฝึกโมเดล Logistic Regression (Logistic_Regression.m) แล้ว
%     -> ได้ไฟล์ malaria_logistic_model.mat
% =====================================================================

clear; clc; close all;

datasetPath = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic';

%% 1) โหลดโมเดล Linear Regression
linearModelFile = fullfile(datasetPath, 'malaria_model.mat');
if ~isfile(linearModelFile)
    error('ไม่พบไฟล์โมเดล Linear Regression: %s\nกรุณารัน Liner.m ก่อน', linearModelFile);
end
linModel = load(linearModelFile);   % .theta, .muX, .sigX, .imgSize

%% 2) โหลดโมเดล Logistic Regression
logisticModelFile = fullfile(datasetPath, 'malaria_logistic_model.mat');
if ~isfile(logisticModelFile)
    error('ไม่พบไฟล์โมเดล Logistic Regression: %s\nกรุณารัน Logistic_Regression.m ก่อน', logisticModelFile);
end
logModel = load(logisticModelFile); % .theta, .muX, .sigX, .imgSize, .threshold

% ทั้งสองโมเดลต้อง resize ภาพขนาดเดียวกันตอนฝึก (ปกติจะเท่ากันอยู่แล้วคือ 50x50)
% ใช้ imgSize จากโมเดล Linear เป็นหลัก (ควรตรงกับ Logistic ด้วย)
imgSize = linModel.imgSize;

%% 3) คลิกเลือกไฟล์ภาพที่ต้องการทดสอบ
[fileName, filePath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.bmp;*.tif', 'ไฟล์ภาพ (*.png, *.jpg, *.jpeg, *.bmp, *.tif)'}, ...
    'เลือกภาพเซลล์ที่ต้องการเปรียบเทียบผล', ...
    datasetPath);

if isequal(fileName, 0)
    disp('ยกเลิกการเลือกไฟล์ภาพ');
    return;
end

imagePath = fullfile(filePath, fileName);
img = imread(imagePath);

%% 4) สกัดฟีเจอร์ (ใช้ฟังก์ชันเดียวกันสำหรับทั้ง 2 โมเดล)
feat = getCellFeatures(img, imgSize);

labelStr = {'เซลล์ปกติ (Uninfected)', 'เซลล์ติดเชื้อมาลาเรีย (Parasitized)'};

%% 5) ทำนายด้วย Linear Regression
featNormLin = (feat - linModel.muX) ./ linModel.sigX;
featAugLin  = [1, featNormLin];
yhatLin     = featAugLin * linModel.theta;
predLin     = double(yhatLin >= 0.5);

%% 6) ทำนายด้วย Logistic Regression
featNormLog = (feat - logModel.muX) ./ logModel.sigX;
featAugLog  = [1, featNormLog];
sigmoid     = @(z) 1 ./ (1 + exp(-z));
probLog     = sigmoid(featAugLog * logModel.theta);
predLog     = double(probLog >= logModel.threshold);

%% 7) แสดงผลเปรียบเทียบใน Command Window
fprintf('\n===================================================\n');
fprintf('        เปรียบเทียบผลการทำนาย: Linear vs Logistic\n');
fprintf('===================================================\n');
fprintf('ไฟล์ภาพ : %s\n\n', imagePath);

fprintf('--- Linear Regression ---\n');
fprintf('Raw score  : %.4f\n', yhatLin);
fprintf('ผลทำนาย    : %s\n\n', labelStr{predLin + 1});

fprintf('--- Logistic Regression ---\n');
fprintf('Probability: %.4f (%.1f%%)\n', probLog, probLog*100);
fprintf('ผลทำนาย    : %s\n\n', labelStr{predLog + 1});

if predLin == predLog
    fprintf(">>> ผลทั้งสองโมเดล 'ตรงกัน'\n");
else
    fprintf(">>> ผลทั้งสองโมเดล 'ไม่ตรงกัน' <<< ควรตรวจสอบเพิ่มเติม\n");
end

%% 8) แสดงภาพพร้อมผลเปรียบเทียบแบบ Visual
figure('Name', 'เปรียบเทียบ Linear vs Logistic Regression', 'NumberTitle', 'off');
imshow(img);

if predLin == predLog
    boxColor = [0 0.6 0];   % เขียว = ตรงกัน
else
    boxColor = [0.85 0 0];  % แดง = ไม่ตรงกัน
end

titleText = sprintf(['Linear   : %s (score=%.2f)\n' ...
                      'Logistic : %s (prob=%.2f)'], ...
    labelStr{predLin+1}, yhatLin, labelStr{predLog+1}, probLog);

title(titleText, 'Color', boxColor, 'FontSize', 11, 'FontWeight', 'bold');


%% =====================================================================
%  ฟังก์ชันสกัดฟีเจอร์จากภาพเซลล์ (ต้องเหมือนกับตอนฝึกโมเดลทั้งสองตัวทุกประการ)
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