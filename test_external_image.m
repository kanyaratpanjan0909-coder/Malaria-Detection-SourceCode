%% =====================================================================
%  ทดสอบภาพจากภายนอก (นอกชุด dataset) ด้วยโมเดล Linear Regression ที่ฝึกไว้แล้ว
%  =====================================================================
%  วิธีใช้งาน:
%   1) รันสคริปต์ฝึกโมเดลก่อน (ต้องมีบรรทัด save('malaria_model.mat', ...)
%      ต่อท้ายสคริปต์ฝึกโมเดล ตามที่แนะนำด้านล่าง)
%   2) รันสคริปต์นี้ จะมีหน้าต่างเปิดขึ้นมาให้คลิกเลือกไฟล์ภาพที่ต้องการตรวจสอบ
%      (ไม่ต้องพิมพ์ path เอง)
% =====================================================================

clear; clc; close all;

%% 1) โหลดโมเดลที่ฝึกไว้แล้ว (theta, mu, sigma สำหรับ normalize, imgSize)
modelFile = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic\malaria_model.mat';   % ต้องตรงกับ path ที่ตอนฝึกโมเดล save ไว้

if ~isfile(modelFile)
    error(['ไม่พบไฟล์โมเดล %s\nกรุณารันสคริปต์ฝึกโมเดลก่อน ' ...
           'และเพิ่มบรรทัด save(''malaria_model.mat'', ''theta'', ''muX'', ''sigX'', ''imgSize'');' ...
           ' ต่อท้ายสคริปต์ฝึกโมเดล'], modelFile);
end

loaded = load(modelFile);
theta   = loaded.theta;
muX     = loaded.muX;
sigX    = loaded.sigX;
imgSize = loaded.imgSize;

%% 2) เปิดหน้าต่างให้คลิกเลือกไฟล์ภาพที่ต้องการทดสอบ (ไม่ต้องพิมพ์ path เอง)
startFolder = 'C:\Users\NB\Desktop\Malaria_Project\Project\Logistic';  % โฟลเดอร์เริ่มต้นตอนเปิดหน้าต่างเลือกไฟล์

[fileName, filePath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.bmp;*.tif', 'ไฟล์ภาพ (*.png, *.jpg, *.jpeg, *.bmp, *.tif)'}, ...
    'เลือกภาพเซลล์ที่ต้องการตรวจสอบ', ...
    startFolder);

if isequal(fileName, 0)
    % ผู้ใช้กด Cancel ในหน้าต่างเลือกไฟล์
    disp('ยกเลิกการเลือกไฟล์ภาพ');
    return;
end

imagePath = fullfile(filePath, fileName);

if ~isfile(imagePath)
    error('ไม่พบไฟล์ภาพ: %s', imagePath);
end

%% 3) อ่านภาพ + สกัดฟีเจอร์ (ใช้ฟังก์ชันเดียวกับตอนฝึกโมเดล)
img = imread(imagePath);
feat = getCellFeatures(img, imgSize);

%% 4) Normalize ฟีเจอร์ด้วยค่า mu/sigma จากตอนฝึกโมเดล (ห้ามคำนวณใหม่)
featNorm = (feat - muX) ./ sigX;
featAug  = [1, featNorm];   % เพิ่ม bias term

%% 5) ทำนายผล
yhat = featAug * theta;
threshold = 0.5;
predicted = double(yhat >= threshold);

labelStr = {'เซลล์ปกติ (Uninfected)', 'เซลล์ติดเชื้อมาลาเรีย (Parasitized)'};

fprintf('\n=== ผลการตรวจสอบภาพ ===\n');
fprintf('ไฟล์ภาพ     : %s\n', imagePath);
fprintf('ค่าที่โมเดลทำนายได้ (raw score) : %.4f\n', yhat);
fprintf('ผลการวินิจฉัย : %s\n', labelStr{predicted + 1});

%% 6) แสดงภาพพร้อมผลลัพธ์
figure('Name', 'ผลการตรวจสอบภาพภายนอก', 'NumberTitle', 'off');
imshow(img);
if predicted == 1
    titleColor = 'r';
else
    titleColor = [0 0.6 0];
end
title(sprintf('%s\n(raw score = %.3f)', labelStr{predicted+1}, yhat), ...
    'Color', titleColor, 'FontSize', 12, 'FontWeight', 'bold');


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