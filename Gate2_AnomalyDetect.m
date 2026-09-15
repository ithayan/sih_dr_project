function [isReferable, pAbnormal, earlyExit] = Gate2_AnomalyDetect(img)
    % Gate 2: Binary Referable Anomaly Filter
    % Evaluates whether lesions/structural microvascular defects exist.
    % Hard Rule: If p(Abnormal) < 0.30 -> Early Exit (Grade 0 / Healthy).
    
    disp('----------------------------------------------------');
    disp('[Stateflow Gate 2] Anomaly Filter');
    
    if size(img, 3) == 3
        r = double(img(:,:,1));
        g = double(img(:,:,2));
        b = double(img(:,:,3));
    else
        g = double(img);
        r = g;
        b = g;
    end
    
    % Check for micro-hemorrhages/exudate contrast signatures
    red_lesion_pixels = sum((r(:) ./ (g(:) + 1e-4) > 1.8) & (g(:) > 20) & (g(:) < 80));
    bright_exudate_pixels = sum((r(:) > 110) & (g(:) > 95) & (b(:) < 65));
    
    % Calculate empirical probability of abnormality
    anomaly_signal = (red_lesion_pixels / 2500) + (bright_exudate_pixels / 800);
    pAbnormal = 1.0 / (1.0 + exp(-1.8 * (anomaly_signal - 0.7)));
    pAbnormal = min(max(pAbnormal, 0.04), 0.99);
    
    if pAbnormal < 0.30
        isReferable = false;
        earlyExit = true;
        disp(['[GATE 2 EARLY EXIT] p(Abnormal) = ', num2str(pAbnormal, '%.3f'), ' < 0.30 -> Certified Grade 0 (Normal / Healthy).']);
    else
        isReferable = true;
        earlyExit = false;
        disp(['[GATE 2 REFERABLE] p(Abnormal) = ', num2str(pAbnormal, '%.3f'), ' >= 0.30 -> Proceeding to Gate 3 Dual Feature Extraction.']);
    end
end
