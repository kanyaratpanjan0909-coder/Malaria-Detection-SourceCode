% =========================================================================
% สคริปต์ทำ Data Augmentation (เก็บทั้ง "รูปต้นฉบับ" และ 8 เทคนิค ทั้งสีและขาวดำ)
% =========================================================================

clear; clc;

% 1. เลือกโฟลเดอร์ต้นทางที่เก็บรูปภาพ
input_folder = uigetdir('', 'เลือกโฟลเดอร์ต้นทางที่เก็บรูปภาพ');
if input_folder == 0; disp('ยกเลิกการเลือก'); return; end

% 2. เลือกโฟลเดอร์หลักปลายทาง
base_output_folder = uigetdir('', 'เลือกโฟลเดอร์หลักสำหรับบันทึกผลลัพธ์');
if base_output_folder == 0; disp('ยกเลิกการเลือก'); return; end

% สร้างโฟลเดอร์ย่อยสำหรับเก็บภาพสีและขาวดำแยกกัน
color_output = fullfile(base_output_folder, 'Color_Augmented');
bw_output = fullfile(base_output_folder, 'BW_Augmented');

if ~exist(color_output, 'dir'); mkdir(color_output); end
if ~exist(bw_output, 'dir'); mkdir(bw_output); end

% 3. ค้นหาไฟล์ภาพทั้งหมด (รองรับ .jpg, .png ทั้งในโฟลเดอร์หลักและโฟลเดอร์ย่อย)
image_files = [dir(fullfile(input_folder, '**', '*.jpg')); ...
               dir(fullfile(input_folder, '**', '*.png')); ...
               dir(fullfile(input_folder, '**', '*.JPG')); ...
               dir(fullfile(input_folder, '**', '*.PNG'))];
           
num_images = length(image_files);
if num_images == 0
    disp('❌ ไม่พบไฟล์รูปภาพในโฟลเดอร์ที่เลือก');
    return;
end

disp(['พบรูปภาพทั้งหมด ', num2str(num_images), ' รูป กำลังประมวลผล (รวมรูปต้นฉบับ)...']);

% 4. วนลูปประมวลผลทีละภาพ
for i = 1:num_images
    base_filename = image_files(i).name;
    full_path = fullfile(image_files(i).folder, base_filename);
    
    img_raw = imread(full_path);
    [h, w, c] = size(img_raw);
    [~, name, ext] = fileparts(base_filename);
    
    % จัดการภาพสี (แปลงเป็นภาพสี 3 แชนแนลเสมอ)
    if c == 3
        img_color = img_raw;
    else
        img_color = cat(3, img_raw, img_raw, img_raw);
    end
    
    % จัดการภาพขาวดำ (Grayscale)
    if c == 3
        img_bw = rgb2gray(img_raw);
    else
        img_bw = img_raw;
    end
    
    % ---------------------------------------------------------
    % 0. บันทึก "รูปต้นฉบับ" ลงไปในโฟลเดอร์ผลลัพธ์ด้วย
    % ---------------------------------------------------------
    imwrite(img_color, fullfile(color_output, [name, '_C_original', ext]));
    imwrite(img_bw, fullfile(bw_output, [name, '_BW_original', ext]));
    
    % ---------------------------------------------------------
    % รัน 8 เทคนิคสำหรับ "ภาพสี"
    % ---------------------------------------------------------
    imwrite(flip(img_color, 2), fullfile(color_output, [name, '_C_flipH', ext]));
    imwrite(flip(img_color, 1), fullfile(color_output, [name, '_C_flipV', ext]));
    
    crop_h = round(h * 0.8); crop_w = round(w * 0.8);
    sy = randi([1, max(1, h - crop_h + 1)]); sx = randi([1, max(1, w - crop_w + 1)]);
    imwrite(imresize(img_color(sy:sy+crop_h-1, sx:sx+crop_w-1, :), [h, w]), fullfile(color_output, [name, '_C_crop', ext]));
    
    angle = randi([1, 359]);
    imwrite(imrotate(img_color, angle, 'bilinear', 'crop'), fullfile(color_output, [name, '_C_rot', num2str(angle), ext]));
    
    tx = randi([-30, 30]); ty = randi([-30, 30]);
    imwrite(imtranslate(img_color, [tx, ty], 'FillValues', 0), fullfile(color_output, [name, '_C_trans', ext]));
    
    bright = 0.8 + rand() * 0.4;
    imwrite(im2uint8(im2double(img_color) * bright), fullfile(color_output, [name, '_C_color', ext]));
    
    sigma = 1 + rand() * 1.5;
    imwrite(imgaussfilt(img_color, sigma), fullfile(color_output, [name, '_C_blur', ext]));
    
    img_ce = img_color;
    eh = round(h * 0.25); ew = round(w * 0.25);
    ey = randi([1, max(1, h - eh + 1)]); ex = randi([1, max(1, w - ew + 1)]);
    img_ce(ey:ey+eh-1, ex:ex+ew-1, :) = 128;
    imwrite(img_ce, fullfile(color_output, [name, '_C_erase', ext]));
    
    img_cv = flip(img_color, 1);
    imwrite(im2uint8(0.5 * im2double(img_color) + 0.5 * im2double(img_cv)), fullfile(color_output, [name, '_C_mix', ext]));

    % ---------------------------------------------------------
    % รัน 8 เทคนิคสำหรับ "ภาพขาวดำ"
    % ---------------------------------------------------------
    imwrite(flip(img_bw, 2), fullfile(bw_output, [name, '_BW_flipH', ext]));
    imwrite(flip(img_bw, 1), fullfile(bw_output, [name, '_BW_flipV', ext]));
    
    imwrite(imresize(img_bw(sy:sy+crop_h-1, sx:sx+crop_w-1), [h, w]), fullfile(bw_output, [name, '_BW_crop', ext]));
    imwrite(imrotate(img_bw, angle, 'bilinear', 'crop'), fullfile(bw_output, [name, '_BW_rot', num2str(angle), ext]));
    imwrite(imtranslate(img_bw, [tx, ty], 'FillValues', 0), fullfile(bw_output, [name, '_BW_trans', ext]));
    imwrite(im2uint8(min(1, im2double(img_bw) * bright)), fullfile(bw_output, [name, '_BW_color', ext]));
    imwrite(imgaussfilt(img_bw, sigma), fullfile(bw_output, [name, '_BW_blur', ext]));
    
    img_be = img_bw;
    img_be(ey:ey+eh-1, ex:ex+ew-1) = 128;
    imwrite(img_be, fullfile(bw_output, [name, '_BW_erase', ext]));
    
    img_bv = flip(img_bw, 1);
    imwrite(im2uint8(0.5 * im2double(img_bw) + 0.5 * im2double(img_bv)), fullfile(bw_output, [name, '_BW_mix', ext]));

    disp(['ประมวลผลสำเร็จรวมต้นฉบับ: รูปที่ ', num2str(i), '/', num2str(num_images), ' (', name, ext, ')']);
end

disp('--- 🎉 ทำ Data Augmentation และบันทึกรูปต้นฉบับเสร็จสมบูรณ์แล้วครับ! ---');