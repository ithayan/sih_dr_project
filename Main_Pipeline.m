function [finalResult, explanation, report] = Main_Pipeline(imageInput)
    % NetraX: 4-Gate Stateflow Retinal Telemedicine Triage Pipeline (SIH26038)
    % Hard-gated clinical architecture:
    %   Gate 1: Image Quality Assessment (Hard Block if score < 0.70)
    %   Gate 2: Binary Referable Anomaly Filter (Early Exit if p < 0.30)
    %   Gate 3: Dual YOLO26 Nano (Branch A) & Vascular GNN (Branch B) Extraction
    %   Gate 4: KAN Severity Staging & Platt Confidence Escalation
    
    disp('========================================================================');
    disp('NETRAX CLINICAL AI SCREENING WORKSTATION: STATEFLOW PIPELINE EXECUTION');
    disp('========================================================================');
    
    report = struct();
    report.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    % Read image if path string provided
    if ischar(imageInput) || isstring(imageInput)
        report.imagePath = char(imageInput);
        img = imread(imageInput);
    else
        report.imagePath = 'In-Memory Matrix';
        img = imageInput;
    end
    
    %% ---------------------------------------------------------------------
    %% GATE 1: Image Quality Assessment (HARD BLOCK)
    %% ---------------------------------------------------------------------
    [isGradable, qualityScore, processedImg, gate1Msg] = Gate1_QualityCheck(img);
    report.gate1_qualityScore = qualityScore;
    report.gate1_passed = isGradable;
    report.gate1_status = gate1Msg;
    
    if ~isGradable
        finalResult = 'Image Quality Insufficient - Recapture';
        explanation = sprintf('HALT PIPELINE: Focus/illumination score (%.3f) is below diagnostic threshold (0.70). Downstream networks locked.', qualityScore);
        report.finalDiagnosis = finalResult;
        report.triageRouting = 'HARD BLOCK: RETAKE REQUIRED';
        disp(['Pipeline Status: ' explanation]);
        disp('========================================================================');
        return;
    end
    
    %% ---------------------------------------------------------------------
    %% GATE 2: Referable Anomaly Filter
    %% ---------------------------------------------------------------------
    [isReferable, pAbnormal, earlyExit] = Gate2_AnomalyDetect(processedImg);
    report.gate2_pAbnormal = pAbnormal;
    report.gate2_referable = isReferable;
    
    if earlyExit
        finalResult = 'Grade 0: Normal / Healthy Retina';
        explanation = sprintf('EARLY EXIT: Certified healthy baseline (p_abnormal = %.3f < 0.30). No referral required.', pAbnormal);
        report.finalDiagnosis = finalResult;
        report.triageRouting = 'CERTIFIED AUTONOMOUS EDGE TRIAGE';
        disp(['Pipeline Status: ' explanation]);
        disp('========================================================================');
        return;
    end
    
    %% ---------------------------------------------------------------------
    %% GATE 3: Dual Semantic YOLO26 Nano & Topological Vascular GNN Extraction
    %% ---------------------------------------------------------------------
    [yoloFeat, gnnFeat, lesionTypology, vascBiomarkers, yoloDetections] = Gate3_LesionIdentify(processedImg);
    report.yoloFeatures = yoloFeat;
    report.gnnFeatures = gnnFeat;
    report.lesions = lesionTypology;
    report.vascularBiomarkers = vascBiomarkers;
    report.yolo26Detections = yoloDetections;
    
    %% ---------------------------------------------------------------------
    %% GATE 4: KAN Severity Staging & Confidence Routing
    %% ---------------------------------------------------------------------
    [sevGrade, sevText, confidence, isEscalated, triageAction] = Gate4_SeverityGrade(yoloFeat, gnnFeat, lesionTypology, vascBiomarkers);
    report.severityGrade = sevGrade;
    report.severityText = sevText;
    report.confidence = confidence;
    report.isEscalated = isEscalated;
    report.triageRouting = triageAction;
    
    finalResult = sevText;
    if isEscalated
        explanation = sprintf('HIGH UNCERTAINTY: Scan routed to Ophthalmologist Review Queue (Confidence = %.1f%%).', confidence * 100);
    else
        explanation = sprintf('DIAGNOSIS CERTIFIED: %s validated via YOLO26-GNN-KAN fusion (Confidence = %.1f%%).', sevText, confidence * 100);
    end
    report.finalDiagnosis = finalResult;
    
    disp('----------------------------------------------------');
    disp(['FINAL TRIAGE OUTCOME: ' finalResult]);
    disp(['CLINICAL ROUTING:     ' triageAction]);
    disp(['EXPLANATION:          ' explanation]);
    disp('========================================================================');
end
