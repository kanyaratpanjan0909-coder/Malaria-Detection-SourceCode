function data = read_images_from_folder(folder_name)
% ฟังก์ชันสำหรับอ่านรูปภาพทั้งหมดในโฟลเดอร์ และคืนค่าเป็น Matrix (Nx3)
% โดยแต่ละแถวคือค่าเฉลี่ยของสี [Red, Green, Blue] ของแต่ละรูป

    % รองรับนามสกุลไฟล์ทั่วไป (jpg, jpeg, png, bmp, tif)
    filePatterns = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif'};
    files = [];
    
    % รวบรวมรายชื่อไฟล์ทั้งหมด
    for i = 1:length(filePatterns)
        files = [files; dir(fullfile(folder_name, filePatterns{i}))];
    end

    if isempty(files)
        warning('ไม่พบรูปภาพในโฟลเดอร์: %s', folder_name);
        data = [];
        return;
    end

    % จองพื้นที่ตัวแปรเพื่อความเร็ว
    data = zeros(length(files), 3); 

    for k = 1:length(files)
        baseFileName = files(k).name;
        fullFileName = fullfile(folder_name, baseFileName);
        
        try
            img = imread(fullFileName);
            
            % ถ้าเป็นภาพขาวดำ (มีแค่ 2 มิติ) ให้แปลงเป็น 3 มิติ (RGB)
            if size(img, 3) == 1
                img = cat(3, img, img, img);
            end
            
            % คำนวณค่าเฉลี่ยของสี R, G, B ทั้งภาพ
            r = double(img(:,:,1));
            g = double(img(:,:,2));
            b = double(img(:,:,3));
            
            avgR = mean(r(:));
            avgG = mean(g(:));
            avgB = mean(b(:));
            
            % บันทึกค่าลงใน Matrix
            data(k, :) = [avgR, avgG, avgB];
            
        catch ME
            warning('อ่านไฟล์ %s ไม่สำเร็จ: %s', baseFileName, ME.message);
        end
    end
end