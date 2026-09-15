% setup_mock_models.m
% Initializes mock deep learning models or structures for the pipeline

function models = setup_mock_models()
    disp('Initializing Mock Models for Pipeline Demo...');
    % In a real scenario, this would be: models.gate1 = load('gate1_model.mat');
    
    models = struct();
    
    % Gate 1: Gradability Model (Binary)
    models.gate1_net = 'Mock_ResNet18_Gate1';
    models.gate1_classes = {'Gradable', 'Ungradable'};
    
    % Gate 2: Anomaly Detection Model (Binary)
    models.gate2_net = 'Mock_ResNet50_Gate2';
    models.gate2_classes = {'Normal', 'Abnormal'};
    
    % Gate 3: Lesion Detection Model (Multi-label or Object Detection)
    models.gate3_net = 'Mock_YOLOv4_Gate3';
    models.gate3_classes = {'Microaneurysm', 'Hemorrhage', 'Exudate'};
    
    % Gate 4: Severity Grading Model (Multi-class)
    models.gate4_net = 'Mock_EfficientNet_Gate4';
    models.gate4_classes = {'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'};
    
    disp('Mock models initialized successfully.');
end
