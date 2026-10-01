  clc; clear; close all;
rng(1);

%% 1. Load Data
disp('Loading dataset (Malaria Cells)...');

try
    % Load images and normalize pixel values to [0, 1]
    % เช็คชื่อโฟลเดอร์ให้ตรงกับที่มีในเครื่อง
    train_para   = double(read_images_from_folder('Train Parasitized')) / 255;
    train_uninf  = double(read_images_from_folder('Train Uninfected')) / 255;
    test_para    = double(read_images_from_folder('Test Parasitized')) / 255;
    test_uninf   = double(read_images_from_folder('Test Uninfected')) / 255;
catch ME
    errordlg('Error: Image folders not found. Please check folder names.', 'Load Error');
    error(ME.message);
end

% Construct Training Data
X_train_raw = [train_para; train_uninf];
% Labels: 1 = Parasitized (ติดเชื้อ), 0 = Uninfected (ไม่ติดเชื้อ)
y_train = [ones(size(train_para, 1), 1); zeros(size(train_uninf, 1), 1)];

% Construct Testing Data
X_test_raw = [test_para; test_uninf];
y_actual = [ones(size(test_para, 1), 1); zeros(size(test_uninf, 1), 1)];

%% 2. Train Model (Logistic Regression)
disp('Training Logistic Regression Model...');

% ใช้ glmfit สำหรับ Logistic Regression
% 'binomial' = ข้อมูลมี 2 คลาส (0,1)
% 'link','logit' = ใช้ Sigmoid function
% หมายเหตุ: glmfit จะเพิ่มค่า Bias (Column of Ones) ให้เองอัตโนมัติ ไม่ต้องใส่เพิ่ม
w = glmfit(X_train_raw, y_train, 'binomial', 'link', 'logit');

%% 3. Prediction & Evaluation

% ทำนายผลด้วย Logistic Regression Model
% w(1) คือค่า Bias (Intercept)
% w(2:end) คือค่า Weight ของ R, G, B
z = w(1) + (X_test_raw * w(2:end));

% คำนวณ Sigmoid (Probability)
y_prob = 1 ./ (1 + exp(-z));

% Classification Rule
y_pred_class = zeros(size(y_prob));

for i = 1:length(y_prob)
    % Threshold at 0.5
    if y_prob(i) >= 0.5
        y_pred_class(i) = 1; % Parasitized
    else
        y_pred_class(i) = 0; % Uninfected
    end
end

% Calculate Regression Metrics
MSE  = mean((y_actual - y_prob).^2);
RMSE = sqrt(MSE);
MAE  = mean(abs(y_actual - y_prob));

% Calculate Classification Metrics
TP = sum((y_pred_class == 1) & (y_actual == 1));
TN = sum((y_pred_class == 0) & (y_actual == 0));
FP = sum((y_pred_class == 1) & (y_actual == 0));
FN = sum((y_pred_class == 0) & (y_actual == 1));

Accuracy  = (TP + TN) / length(y_actual);
Precision = TP / (TP + FP);
Recall    = TP / (TP + FN);
F1_Score  = 2 * (Precision * Recall) / (Precision + Recall);

% Handle NaN values
if isnan(Precision), Precision = 0; end
if isnan(Recall), Recall = 0; end
if isnan(F1_Score), F1_Score = 0; end

%% 4. Display Results
fprintf('\n--------------------------------------------------\n');
fprintf('Malaria Detection Model Report (Logistic Regression)\n');
fprintf('--------------------------------------------------\n');

% Weights
fprintf('[1] Model Weights\n');
fprintf('    w0 (Bias) : %.4f\n', w(1));
fprintf('    w1 (Red)  : %.4f\n', w(2));
fprintf('    w2 (Green): %.4f\n', w(3));
fprintf('    w3 (Blue) : %.4f\n', w(4));

% Average RGB
fprintf('\n[2] Average RGB Values (Normalized)\n');
fprintf('    Parasitized (Train): R=%.4f, G=%.4f, B=%.4f\n', mean(train_para));
fprintf('    Uninfected  (Train): R=%.4f, G=%.4f, B=%.4f\n', mean(train_uninf));

% Error Metrics
fprintf('\n[3] Error Metrics\n');
fprintf('    MSE  : %.4f\n', MSE);
fprintf('    RMSE : %.4f\n', RMSE);
fprintf('    MAE  : %.4f\n', MAE);

% Classification Metrics
fprintf('\n[4] Classification Metrics\n');
fprintf('    Accuracy  : %.2f%%\n', Accuracy * 100);
fprintf('    Precision : %.4f\n', Precision);
fprintf('    Recall    : %.4f\n', Recall);
fprintf('    F1-Score  : %.4f\n', F1_Score);

% Confusion Matrix
fprintf('\n[5] Confusion Matrix\n');
fprintf('    TP (Parasitized correctly ID): %d\n', TP);
fprintf('    TN (Uninfected correctly ID) : %d\n', TN);
fprintf('    FP (False Alarm)             : %d\n', FP);
fprintf('    FN (Missed Detection)        : %d\n', FN);
fprintf('--------------------------------------------------\n');

%% 5. Manual Test
disp('Starting manual test loop...');

while true
    [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.JPG;*.bmp;*.tif'}, 'Select Cell Image');
    if isequal(file, 0), break; end
    
    fullFileName = fullfile(path, file);
    img = imread(fullFileName);
    
    % Process Image
    img_d = double(img) / 255;
    mR = mean(mean(img_d(:,:,1)));
    mG = mean(mean(img_d(:,:,2)));
    mB = mean(mean(img_d(:,:,3)));
    
    % Prediction (Logistic Regression Formula)
    % z = Bias + (wR*R + wG*G + wB*B)
    input_features = [mR, mG, mB]; 
    z = w(1) + (input_features * w(2:end));
    
    % Sigmoid
    raw_score = 1 ./ (1 + exp(-z));
    
    % Decision Logic
    if raw_score >= 0.5
        resTxt = 'PARASITIZED';
        col = 'r'; % Red for Danger/Infected
    else
        resTxt = 'UNINFECTED';
        col = 'g'; % Green for Safe/Clean
    end
    
    % Show Result
    figure(99); imshow(img);
    text(10, 20, sprintf('%s\n(Prob: %.2f)', resTxt, raw_score), ...
        'Color', col, 'FontSize', 18, 'FontWeight', 'bold', 'BackgroundColor', 'w');
    title(sprintf('R=%.2f, G=%.2f, B=%.2f -> Result: %s', mR, mG, mB, resTxt));
    
    fprintf('File: %s | Prob: %.4f | Prediction: %s\n', file, raw_score, resTxt);
end