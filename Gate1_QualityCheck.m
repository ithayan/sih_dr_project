function [isGradable, qualityScore, processedImg, statusMsg] = Gate1_QualityCheck(img)
    % Gate 1: Image Quality Assessment (Hard Block)
    % Evaluates Focus Sharpness (Tenengrad), Illumination Entropy, and Retinal FOV Coverage.
    % Rule: If score < 0.70 -> HALT PIPELINE. Display "Image Quality Insufficient - Recapture".
    %       If score 0.70 - 0.85 -> Route through CLAHE and Denoising.
    
    disp('----------------------------------------------------');
    disp('[Stateflow Gate 1] Image Quality Assessment');
    
    if ischar(img) || isstring(img)
        img = imread(img);
    end
    
    % Convert to double grayscale [0, 1]
    if size(img, 3) == 3
        gray = rgb2gray(im2double(img));
    else
        gray = im2double(img);
    end
    
    % 1. Retinal Field of View (FOV) coverage
    fov_mask = gray > 0.06;
    fov_ratio = mean(fov_mask(:));
    if fov_ratio < 0.18
        isGradable = false;
        qualityScore = 0.20;
        processedImg = img;
        statusMsg = 'Image Quality Insufficient - Recapture';
        fprintf(2, '[GATE 1 HALT] Score = %.3f < 0.70. Non-retinal or empty field.\n', qualityScore);
        return;
    end
    fov_score = min(max(fov_ratio / 0.40, 0.0), 1.0);

    % 2. Sharpness via Tenengrad gradient energy inside retinal tissue
    [gx, gy] = imgradientxy(gray, 'sobel');
    grad_mag = sqrt(gx.^2 + gy.^2);
    retinal_grad = mean(grad_mag(fov_mask));
    sharpness = min(max((retinal_grad - 0.008) / 0.025, 0.0), 1.0);

    % 3. Illumination balance inside retinal tissue
    mean_lum = mean(gray(fov_mask));
    illumination = min(max(1.0 - abs(mean_lum - 0.45) / 0.40, 0.0), 1.0);

    % Composite Quality Score
    composite_q = 0.50 * sharpness + 0.30 * illumination + 0.20 * fov_score;
    qualityScore = min(max(0.35 + 0.60 * composite_q, 0.20), 0.96);
    
    % Hard Gate Evaluation
    if qualityScore < 0.70
        isGradable = false;
        processedImg = img;
        statusMsg = 'Image Quality Insufficient - Recapture';
        fprintf(2, '[GATE 1 HALT] Score = %.3f < 0.70. DOWNSTREAM NETWORKS LOCKED. RECAPTURE SCAN.\n', qualityScore);
    elseif qualityScore <= 0.85
        isGradable = true;
        statusMsg = 'Borderline Quality: Preprocessing with CLAHE & Denoising';
        disp(['[GATE 1 PASSED WITH CLAHE] Score = ', num2str(qualityScore, '%.3f'), ' (Borderline)']);
        % Route through CLAHE on Green / L-channel
        lab = rgb2lab(img);
        lab(:,:,1) = adapthisteq(lab(:,:,1) / 100, 'ClipLimit', 0.02) * 100;
        processedImg = lab2rgb(lab);
        processedImg = imgaussfilt(processedImg, 0.6);
    else
        isGradable = true;
        statusMsg = 'High Quality: Direct Neural Stream';
        processedImg = img;
        disp(['[GATE 1 PASSED] Score = ', num2str(qualityScore, '%.3f'), ' >= 0.85']);
    end
end
