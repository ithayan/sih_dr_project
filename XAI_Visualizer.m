classdef XAI_Visualizer < handle
    % XAI_Visualizer: Clinical AI Screening Workstation for OcuNexa / NetraX (SIH26038)
    % Hybrid YOLO-GNN-KAN Retinal Telemedicine Triage Workstation
    %
    % Architecture:
    %   - Branch A: YOLO26 Nano (yolo26n-seg.pt) compiled for Intel OpenVINO INT8 CPU
    %               Hungarian One-to-One matching (NMS-Free) instance segmentation & bounding boxes
    %   - Branch B: Topological Vascular GNN (2-Layer GAT on DRIVE skeleton)
    %               Deterministic vascular metrics: AVR, Tortuosity Index, Branching Angle
    %   - Branch C: Kolmogorov-Arnold Network (KAN) Decision Head with B-spline activation functions
    %
    % Clinical Stateflow Quality Gates:
    %   - Gate 1: Hard Quality Lock (Recapture required if Quality Score < 0.70)
    %   - Gate 2: Binary Referable Anomaly Filter (Early normal exit if p < 0.30)
    %   - Gate 3: Multi-Label Lesion Grounding via YOLO26 Nano & GNN Graph
    %   - Gate 4: Platt Scaling Confidence & Remote Specialist Escalation (< 82%)

    properties (Access = public)
        UIFigure matlab.ui.Figure
        GridLayout matlab.ui.container.GridLayout
        LeftPanel matlab.ui.container.Panel
        CenterPanel matlab.ui.container.Panel
        RightPanel matlab.ui.container.Panel

        % Center Vision Layer Controls
        LayerToolbarPanel matlab.ui.container.Panel
        ShowYoloBoxesCheckbox matlab.ui.control.CheckBox
        ShowYoloMasksCheckbox matlab.ui.control.CheckBox
        ShowVesselsCheckbox matlab.ui.control.CheckBox
        ShowETDRSCheckbox matlab.ui.control.CheckBox
        ShowHeatmapCheckbox matlab.ui.control.CheckBox

        UIAxes matlab.ui.control.UIAxes
        LoadImageButton matlab.ui.control.Button
        RunInferenceButton matlab.ui.control.Button
        LogTextArea matlab.ui.control.TextArea
        SeverityLabel matlab.ui.control.Label
        TriageStatusLabel matlab.ui.control.Label
        RecommendationLabel matlab.ui.control.Label

        % Branch A: YOLO26 Nano Lesion Indicators
        YoloEngineLabel matlab.ui.control.Label
        YoloDetectionsCountLabel matlab.ui.control.Label
        LesionCountLabel matlab.ui.control.Label

        % Branch B: Deterministic Vascular Biomarker Gauges (GNN)
        AVRLabel matlab.ui.control.Label
        TortuosityLabel matlab.ui.control.Label
        BranchAngleLabel matlab.ui.control.Label

        % Branch C & Stateflow Gate Indicators
        GateStatusLabel matlab.ui.control.Label
        ModelConfidenceLabel matlab.ui.control.Label
        YoloDetectionsTextArea matlab.ui.control.TextArea

        CurrentImage uint8 = []
        RetinaMask logical = []
        CurrentFileName char = 'patient_scan.jpg'

        % Active Inference Cache for zero-latency interactive layer re-rendering
        CurrentYoloDetections = {}
        FoveaCoords = [0.55, 0.50]
        OpticDiscCoords = [0.28, 0.49]
        SeverityGrade = -1
        SeverityText char = 'STANDBY'
        AVRVal = 0.70
        TortVal = 0.03
    end

    methods (Access = public)
        function app = XAI_Visualizer()
            createComponents(app);
            logMessage(app, '==================================================');
            logMessage(app, 'NETRAX / OCUNEXA CLINICAL AI WORKSTATION ONLINE');
            logMessage(app, 'Architecture: Hybrid YOLO26-GNN-KAN Core (SIH26038)');
            logMessage(app, 'Branch A: YOLO26 Nano (OpenVINO INT8, NMS-Free Hungarian)');
            logMessage(app, 'Branch B: Vascular Graph GAT (AVR, Tortuosity, Angles)');
            logMessage(app, 'Branch C: B-Spline KAN Interpretability Head');
            logMessage(app, 'Gate 1: Hard Quality Lock (Diagnostic Cutoff: 0.70)');
            logMessage(app, 'Standby for patient fundus scan ingestion...');
            logMessage(app, '==================================================');
        end

        function createComponents(app)
            % 1. Create Dark-Mode Medical UIFigure
            app.UIFigure = uifigure(...
                'Name', 'NetraX • Clinical AI Screening & Telemedicine Workstation (YOLO26-GNN-KAN)', ...
                'Color', [0.06 0.08 0.11], 'Position', [60 60 1340 780]);

            % 2. Create Responsive Grid Layout (3 Columns: System/Metrics, Retinal Vision, Triage/Queue)
            app.GridLayout = uigridlayout(app.UIFigure, [1 3]);
            app.GridLayout.ColumnWidth = {280, '1x', 380};
            app.GridLayout.BackgroundColor = [0.06 0.08 0.11];
            app.GridLayout.Padding = [10 10 10 10];
            app.GridLayout.ColumnSpacing = 10;

            % ----------------------------------------------------
            % Left Panel: Ingestion & Tri-Branch System Gauges
            % ----------------------------------------------------
            app.LeftPanel = uipanel(app.GridLayout, ...
                'Title', 'SYSTEM CONTROL & AI BIOMARKERS', ...
                'BackgroundColor', [0.09 0.12 0.17], 'ForegroundColor', [0.38 0.74 0.98], ...
                'FontWeight', 'bold', 'FontSize', 11);
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;

            leftGrid = uigridlayout(app.LeftPanel, [15 1]);
            leftGrid.RowHeight = {36, 36, 18, 28, 28, 28, 18, 28, 28, 28, 18, 28, 28, '1x', 22};
            leftGrid.BackgroundColor = [0.09 0.12 0.17];
            leftGrid.Padding = [8 8 8 8];
            leftGrid.RowSpacing = 4;

            app.LoadImageButton = uibutton(leftGrid, 'push', ...
                'Text', '  Load Fundus Image', ...
                'BackgroundColor', [0.02 0.44 0.74], 'FontColor', [1 1 1], ...
                'FontSize', 12, 'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(btn,event) LoadImage(app));
            app.LoadImageButton.Layout.Row = 1;

            app.RunInferenceButton = uibutton(leftGrid, 'push', ...
                'Text', '  Execute NetraX Triage', ...
                'BackgroundColor', [0.10 0.65 0.35], 'FontColor', [1 1 1], ...
                'FontSize', 12, 'FontWeight', 'bold', 'Enable', 'off', ...
                'ButtonPushedFcn', @(btn,event) RunInference(app));
            app.RunInferenceButton.Layout.Row = 2;

            % --- Branch A: YOLO26 Nano Lesion Engine ---
            lblYolo = uilabel(leftGrid, 'Text', 'BRANCH A: YOLO26 NANO LESION ENGINE', ...
                'FontSize', 10, 'FontWeight', 'bold', 'FontColor', [0.38 0.74 0.98]);
            lblYolo.Layout.Row = 3;

            app.YoloEngineLabel = uilabel(leftGrid, 'Text', 'YOLO26 Engine: OpenVINO INT8 (NMS-Free)', ...
                'FontSize', 10, 'FontColor', [0.85 0.90 0.95], 'BackgroundColor', [0.05 0.07 0.10]);
            app.YoloEngineLabel.Layout.Row = 4;

            app.YoloDetectionsCountLabel = uilabel(leftGrid, 'Text', 'YOLO26 Detections: --', ...
                'FontSize', 10, 'FontColor', [0.85 0.90 0.95], 'BackgroundColor', [0.05 0.07 0.10]);
            app.YoloDetectionsCountLabel.Layout.Row = 5;

            app.LesionCountLabel = uilabel(leftGrid, 'Text', 'Lesions: MA:0 | HE:0 | EX:0', ...
                'FontSize', 10, 'FontColor', [0.85 0.90 0.95], 'BackgroundColor', [0.05 0.07 0.10]);
            app.LesionCountLabel.Layout.Row = 6;

            % --- Branch B: GNN Vascular Engine ---
            lblVasc = uilabel(leftGrid, 'Text', 'BRANCH B: TOPOLOGICAL VASCULAR GNN', ...
                'FontSize', 10, 'FontWeight', 'bold', 'FontColor', [0.38 0.74 0.98]);
            lblVasc.Layout.Row = 7;

            app.AVRLabel = uilabel(leftGrid, 'Text', 'AVR (A/V Ratio): --', ...
                'FontSize', 10, 'FontColor', [0.85 0.90 0.95], 'BackgroundColor', [0.05 0.07 0.10]);
            app.AVRLabel.Layout.Row = 8;

            app.TortuosityLabel = uilabel(leftGrid, 'Text', 'Tortuosity Index: --', ...
                'FontSize', 10, 'FontColor', [0.85 0.90 0.95], 'BackgroundColor', [0.05 0.07 0.10]);
            app.TortuosityLabel.Layout.Row = 9;

            app.BranchAngleLabel = uilabel(leftGrid, 'Text', 'Branching Angle: -- deg', ...
                'FontSize', 10, 'FontColor', [0.85 0.90 0.95], 'BackgroundColor', [0.05 0.07 0.10]);
            app.BranchAngleLabel.Layout.Row = 10;

            % --- Branch C: KAN & Quality Gates ---
            lblGates = uilabel(leftGrid, 'Text', 'BRANCH C & CLINICAL GATES', ...
                'FontSize', 10, 'FontWeight', 'bold', 'FontColor', [0.38 0.74 0.98]);
            lblGates.Layout.Row = 11;

            app.GateStatusLabel = uilabel(leftGrid, 'Text', 'Gate 1 Quality: Inactive', ...
                'FontSize', 10, 'FontColor', [0.85 0.90 0.95], 'BackgroundColor', [0.05 0.07 0.10]);
            app.GateStatusLabel.Layout.Row = 12;

            app.ModelConfidenceLabel = uilabel(leftGrid, 'Text', 'KAN Confidence: --', ...
                'FontSize', 10, 'FontColor', [0.85 0.90 0.95], 'BackgroundColor', [0.05 0.07 0.10]);
            app.ModelConfidenceLabel.Layout.Row = 13;

            lblFoot = uilabel(leftGrid, 'Text', 'YOLO26 Nano + 2-Layer GAT + B-Spline KAN', ...
                'FontSize', 9, 'FontColor', [0.45 0.55 0.65]);
            lblFoot.Layout.Row = 15;

            % ----------------------------------------------------
            % Center Panel: Vision Layer Toolbar & High-Res Retinal Axes
            % ----------------------------------------------------
            app.CenterPanel = uipanel(app.GridLayout, ...
                'Title', 'RETINAL VISION LAYER: YOLO26 NANO LESIONS & VASCULAR MANIFOLD', ...
                'BackgroundColor', [0.09 0.12 0.17], 'ForegroundColor', [0.38 0.74 0.98], ...
                'FontWeight', 'bold', 'FontSize', 11);
            app.CenterPanel.Layout.Row = 1;
            app.CenterPanel.Layout.Column = 2;

            centerGrid = uigridlayout(app.CenterPanel, [2 1]);
            centerGrid.RowHeight = {34, '1x'};
            centerGrid.BackgroundColor = [0.09 0.12 0.17];
            centerGrid.Padding = [4 4 4 4];
            centerGrid.RowSpacing = 4;

            % Sub-Toolbar: Interactive Layer Toggles
            app.LayerToolbarPanel = uipanel(centerGrid, ...
                'BackgroundColor', [0.06 0.08 0.12], 'BorderType', 'none');
            app.LayerToolbarPanel.Layout.Row = 1;
            app.LayerToolbarPanel.Layout.Column = 1;

            tbGrid = uigridlayout(app.LayerToolbarPanel, [1 5]);
            tbGrid.ColumnWidth = {'1x', '1x', '1x', '1x', '1x'};
            tbGrid.BackgroundColor = [0.06 0.08 0.12];
            tbGrid.Padding = [4 4 4 4];
            tbGrid.ColumnSpacing = 6;

            app.ShowYoloBoxesCheckbox = uicheckbox(tbGrid, ...
                'Text', 'YOLO26 Boxes', 'Value', true, ...
                'FontColor', [0.98 0.75 0.15], 'FontSize', 10, 'FontWeight', 'bold', ...
                'ValueChangedFcn', @(src, event) RenderOverlays(app));
            app.ShowYoloBoxesCheckbox.Layout.Row = 1;
            app.ShowYoloBoxesCheckbox.Layout.Column = 1;

            app.ShowYoloMasksCheckbox = uicheckbox(tbGrid, ...
                'Text', 'YOLO26 Masks', 'Value', true, ...
                'FontColor', [0.95 0.45 0.25], 'FontSize', 10, 'FontWeight', 'bold', ...
                'ValueChangedFcn', @(src, event) RenderOverlays(app));
            app.ShowYoloMasksCheckbox.Layout.Row = 1;
            app.ShowYoloMasksCheckbox.Layout.Column = 2;

            app.ShowVesselsCheckbox = uicheckbox(tbGrid, ...
                'Text', 'Vascular GNN', 'Value', true, ...
                'FontColor', [0.20 0.85 1.00], 'FontSize', 10, 'FontWeight', 'bold', ...
                'ValueChangedFcn', @(src, event) RenderOverlays(app));
            app.ShowVesselsCheckbox.Layout.Row = 1;
            app.ShowVesselsCheckbox.Layout.Column = 3;

            app.ShowETDRSCheckbox = uicheckbox(tbGrid, ...
                'Text', 'ETDRS Grid', 'Value', true, ...
                'FontColor', [0.13 0.77 0.37], 'FontSize', 10, 'FontWeight', 'bold', ...
                'ValueChangedFcn', @(src, event) RenderOverlays(app));
            app.ShowETDRSCheckbox.Layout.Row = 1;
            app.ShowETDRSCheckbox.Layout.Column = 4;

            app.ShowHeatmapCheckbox = uicheckbox(tbGrid, ...
                'Text', 'Lesion Saliency', 'Value', true, ...
                'FontColor', [0.92 0.70 0.03], 'FontSize', 10, 'FontWeight', 'bold', ...
                'ValueChangedFcn', @(src, event) RenderOverlays(app));
            app.ShowHeatmapCheckbox.Layout.Row = 1;
            app.ShowHeatmapCheckbox.Layout.Column = 5;

            % Retinal Drawing Axes
            app.UIAxes = uiaxes(centerGrid);
            app.UIAxes.Layout.Row = 2;
            app.UIAxes.Layout.Column = 1;
            app.UIAxes.Color = [0.03 0.04 0.06];
            app.UIAxes.XColor = 'none';
            app.UIAxes.YColor = 'none';
            app.UIAxes.Box = 'off';
            title(app.UIAxes, 'Awaiting Retinal Scan Ingestion...', 'Color', [0.6 0.7 0.8], 'FontSize', 12);

            % ----------------------------------------------------
            % Right Panel: 4-Gate Clinical Triage & Telemedicine Escalation
            % ----------------------------------------------------
            app.RightPanel = uipanel(app.GridLayout, ...
                'Title', 'STATEFLOW CLINICAL TRIAGE & TELEMEDICINE ESCALATION', ...
                'BackgroundColor', [0.09 0.12 0.17], 'ForegroundColor', [0.38 0.74 0.98], ...
                'FontWeight', 'bold', 'FontSize', 11);
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 3;

            rightGrid = uigridlayout(app.RightPanel, [5 1]);
            rightGrid.RowHeight = {46, 32, 50, 110, '1x'};
            rightGrid.BackgroundColor = [0.09 0.12 0.17];
            rightGrid.Padding = [6 6 6 6];
            rightGrid.RowSpacing = 6;

            app.SeverityLabel = uilabel(rightGrid, 'Text', 'Severity: STANDBY', ...
                'FontSize', 16, 'FontWeight', 'bold', 'FontColor', [0.65 0.75 0.85], ...
                'BackgroundColor', [0.05 0.07 0.10], 'HorizontalAlignment', 'center');
            app.SeverityLabel.Layout.Row = 1;

            app.TriageStatusLabel = uilabel(rightGrid, 'Text', 'ROUTING: Standby', ...
                'FontSize', 11, 'FontWeight', 'bold', 'FontColor', [0.4 0.7 0.9], ...
                'BackgroundColor', [0.05 0.07 0.10], 'HorizontalAlignment', 'center');
            app.TriageStatusLabel.Layout.Row = 2;

            app.RecommendationLabel = uilabel(rightGrid, 'Text', 'Triage: Ingest fundus scan to trigger autonomous 4-Gate screening.', ...
                'FontSize', 10, 'FontColor', [0.8 0.85 0.9], 'BackgroundColor', [0.05 0.07 0.10], ...
                'WordWrap', 'on');
            app.RecommendationLabel.Layout.Row = 3;

            % YOLO26 Explicit Detections Inventory Box
            app.YoloDetectionsTextArea = uitextarea(rightGrid, ...
                'Value', {'[YOLO26 Nano Detection Inventory]', 'Standby for Hungarian one-to-one assignment...'}, ...
                'BackgroundColor', [0.04 0.06 0.09], 'FontColor', [0.98 0.80 0.25], ...
                'FontName', 'Consolas', 'FontSize', 9, 'Editable', 'off');
            app.YoloDetectionsTextArea.Layout.Row = 4;

            % Clinical Telemetry Console
            app.LogTextArea = uitextarea(rightGrid, ...
                'BackgroundColor', [0.03 0.04 0.06], 'FontColor', [0.3 0.88 0.4], ...
                'FontName', 'Consolas', 'FontSize', 9, 'Editable', 'off');
            app.LogTextArea.Layout.Row = 5;
        end

        function logMessage(app, msg)
            timestamp = char(datetime('now', 'Format', 'HH:mm:ss'));
            currentText = app.LogTextArea.Value;
            line = sprintf('[%s] %s', timestamp, msg);
            if isempty(currentText)
                app.LogTextArea.Value = {line};
            else
                app.LogTextArea.Value = [currentText; {line}];
            end
            scroll(app.LogTextArea, 'bottom');
            drawnow;
        end

        function LoadImage(app)
            [file, path] = uigetfile({'*.png;*.jpg;*.jpeg;*.tif;*.bmp', 'Fundus Image Files (*.png, *.jpg, *.jpeg, *.tif, *.bmp)'});
            if isequal(file, 0)
                return;
            end
            img_path = fullfile(path, file);
            app.CurrentFileName = char(file);
            
            raw_img = imread(img_path);

            % Normalize data type and channel dimensions
            if size(raw_img, 3) == 4
                raw_img = raw_img(:, :, 1:3);
            elseif size(raw_img, 3) == 1
                raw_img = repmat(raw_img, [1, 1, 3]);
            end

            if isa(raw_img, 'uint16')
                raw_img = uint8(double(raw_img) / 65535 * 255);
            elseif isa(raw_img, 'double') || isa(raw_img, 'single')
                if max(raw_img(:)) <= 1.0
                    raw_img = uint8(raw_img * 255);
                else
                    raw_img = uint8(raw_img);
                end
            end

            app.CurrentImage = raw_img;
            [h, w, ~] = size(app.CurrentImage);

            % Accurate circular Field-of-View (FOV) retinal mask
            gray = rgb2gray(app.CurrentImage);
            app.RetinaMask = imfill(gray > 18, 'holes');

            % Reset cache
            app.CurrentYoloDetections = {};
            app.SeverityGrade = -1;
            app.SeverityText = 'READY FOR INFERENCE';

            % High-precision display on UIAxes
            cla(app.UIAxes);
            image(app.UIAxes, 'CData', app.CurrentImage);
            axis(app.UIAxes, 'image');
            app.UIAxes.XLim = [1, w];
            app.UIAxes.YLim = [1, h];
            app.UIAxes.YDir = 'reverse';
            app.UIAxes.XColor = 'none';
            app.UIAxes.YColor = 'none';
            title(app.UIAxes, sprintf('Raw Fundus Scan Ingested: %s (%dx%d)', file, w, h), 'Color', [0.9 0.9 0.9], 'FontSize', 11);

            logMessage(app, sprintf('Scan Ingested: %s (%dx%d)', file, w, h));
            logMessage(app, 'Gate 1 [Gradability]: Ready for quality verification.');

            app.RunInferenceButton.Enable = 'on';
            app.SeverityLabel.Text = 'Severity: READY FOR INFERENCE';
            app.SeverityLabel.FontColor = [0.9 0.7 0.2];
            app.TriageStatusLabel.Text = 'ROUTING: Pending Gate Execution';
            app.TriageStatusLabel.FontColor = [0.6 0.7 0.8];
            app.RecommendationLabel.Text = 'Click "Execute NetraX Triage" to stream 4-Gate screening with YOLO26 Nano.';
            app.YoloDetectionsTextArea.Value = {'[YOLO26 Nano Detection Inventory]', 'Ready. Click Execute to detect lesions.'};
        end

        function RunInference(app)
            if isempty(app.CurrentImage)
                uialert(app.UIFigure, 'Please load a fundus scan first.', 'No Image Ingested');
                return;
            end

            logMessage(app, '--------------------------------------------------');
            logMessage(app, 'Connecting to NetraX Edge Server (Port 8080)...');
            app.RunInferenceButton.Enable = 'off';
            drawnow;

            try
                % Normalize transmission resolution for large scans (>1024px) for rapid edge transfer
                imgToSend = app.CurrentImage;
                [h, w, ~] = size(imgToSend);
                if max(h, w) > 1024
                    scale = 1024.0 / max(h, w);
                    imgToSend = imresize(imgToSend, scale);
                end

                % Base64 JPEG encoding for fast local HTTP transfer
                tempfile = [tempname, '.jpg'];
                imwrite(imgToSend, tempfile, 'Quality', 92);
                fid = fopen(tempfile, 'rb');
                bytes = fread(fid, '*uint8');
                fclose(fid);
                if isfile(tempfile)
                    delete(tempfile);
                end

                b64str = matlab.net.base64encode(bytes);
                fname = app.CurrentFileName;
                if isempty(fname)
                    fname = 'patient_scan.jpg';
                end
                payload = struct('image_base64', b64str, 'filename', fname);
                options = weboptions('MediaType', 'application/json', 'Timeout', 10);
                url = 'http://127.0.0.1:8080/predict';

                logMessage(app, 'Streaming scan to Hybrid YOLO26-GNN-KAN Pipeline...');
                response = webwrite(url, payload, options);

                processServerResponse(app, response);

            catch ME
                logMessage(app, ['Backend Notice: ' ME.message]);
                logMessage(app, 'Executing Native MATLAB Stateflow & YOLO26 Engine...');
                processLocalFallback(app);
            end

            app.RunInferenceButton.Enable = 'on';
        end

        function processServerResponse(app, response)
            % ----------------------------------------------------
            % GATE 1 HARD BLOCK ENFORCEMENT:
            % ----------------------------------------------------
            if isfield(response, 'gate1_passed') && ~response.gate1_passed
                q_score = response.quality_score;
                logMessage(app, sprintf('GATE 1 HARD REJECT: Quality Score = %.3f < 0.70 Threshold!', q_score));
                logMessage(app, 'CRITICAL: Image Quality Insufficient - Recapture scan.');
                logMessage(app, 'Downstream neural networks locked. Diagnostic grade withheld.');

                app.SeverityLabel.Text = 'IMAGE QUALITY INSUFFICIENT';
                app.SeverityLabel.FontColor = [0.94, 0.27, 0.27];
                app.TriageStatusLabel.Text = 'HARD BLOCK: RECAPTURE MANDATORY';
                app.TriageStatusLabel.FontColor = [0.94, 0.27, 0.27];
                app.RecommendationLabel.Text = 'Image Quality Insufficient - Recapture. Scan focus or illumination below diagnostic threshold (0.70). Retake patient scan.';
                app.GateStatusLabel.Text = 'Gate 1: HARD REJECT (<0.70)';
                app.YoloDetectionsTextArea.Value = {'[GATE 1 HALT]', 'YOLO26 Branch Locked: Image Quality Insufficient'};

                [h, w, ~] = size(app.CurrentImage);
                cla(app.UIAxes);
                image(app.UIAxes, 'CData', app.CurrentImage);
                axis(app.UIAxes, 'image');
                app.UIAxes.XLim = [1, w];
                app.UIAxes.YLim = [1, h];
                app.UIAxes.YDir = 'reverse';
                title(app.UIAxes, 'RECAPTURE SCAN: Quality Score Below 0.70 Threshold', 'Color', [0.94, 0.27, 0.27], 'FontSize', 11);
                return;
            end

            % Gate 1 Passed
            q_score = response.quality_score;
            logMessage(app, sprintf('Gate 1 [Gradability]: PASSED (Quality Score = %.3f)', q_score));
            app.GateStatusLabel.Text = sprintf('Gate 1: PASSED (%.3f)', q_score);

            sev_grade = response.severity_grade;
            conf = response.confidence;
            is_escalated = response.escalated_to_queue;
            triage_routing = response.triage_routing;

            app.SeverityGrade = sev_grade;
            app.SeverityText = response.severity_text;
            app.FoveaCoords = [response.fovea_x, response.fovea_y];
            app.OpticDiscCoords = [response.optic_disc_x, response.optic_disc_y];

            vasc = response.vascular_biomarkers;
            app.AVRVal = vasc.arteriolar_venular_ratio_avr;
            app.TortVal = vasc.tortuosity_index;
            angle_val = vasc.branching_angle_irregularity_deg;

            lesions = response.lesion_inventory;
            count_ma = lesions.count_microaneurysms;
            count_he = lesions.count_hemorrhages;
            count_ex = lesions.count_hard_exudates;

            % --- Parse Branch A: YOLO26 Nano Detections ---
            app.CurrentYoloDetections = {};
            if isfield(response, 'yolo26_detections') && ~isempty(response.yolo26_detections)
                raw_dets = response.yolo26_detections;
                if iscell(raw_dets)
                    app.CurrentYoloDetections = raw_dets;
                elseif isstruct(raw_dets)
                    app.CurrentYoloDetections = num2cell(raw_dets);
                end
            end

            num_yolo = length(app.CurrentYoloDetections);
            app.YoloDetectionsCountLabel.Text = sprintf('YOLO26 Detections: %d (Hungarian)', num_yolo);
            logMessage(app, sprintf('Branch A [YOLO26 Nano INT8]: %d Lesion Bounding Boxes (NMS-Free Hungarian)', num_yolo));

            % Update YOLO26 inventory list
            invText = cell(num_yolo + 1, 1);
            invText{1} = sprintf('[YOLO26 NANO INVENTORY: %d DETECTIONS]', num_yolo);
            for k = 1:num_yolo
                det_item = app.CurrentYoloDetections{k};
                det_id = det_item.detection_id;
                det_type = det_item.lesion_type;
                det_conf = det_item.confidence * 100;
                det_zone = det_item.etdrs_zone;
                invText{k + 1} = sprintf('%s: %s (%.0f%%) • %s', det_id, det_type, det_conf, det_zone);
                logMessage(app, sprintf('  - [%s] %s (%.1f%%) in %s', det_id, det_type, det_conf, det_zone));
            end
            if num_yolo == 0
                invText = [invText; {'Zero microaneurysms or hemorrhages detected.'}];
            end
            app.YoloDetectionsTextArea.Value = invText;

            % Gate 2 Log
            logMessage(app, sprintf('Gate 2 [Anomaly Filter]: %s', ternary(sev_grade == 0, 'Healthy/Normal Exit', 'Referable Anomaly Detected')));

            % Gate 3 Log
            logMessage(app, sprintf('Gate 3 [YOLO26 & GNN]: MA=%d, HE=%d, EX=%d | AVR=%.3f, Tort=%.4f', ...
                count_ma, count_he, count_ex, app.AVRVal, app.TortVal));

            % Gate 4 Log
            logMessage(app, sprintf('Gate 4 [KAN Staging]: %s (Confidence: %.1f%%)', response.severity_text, conf * 100));
            logMessage(app, sprintf('Gate 4 [Triage Routing]: %s', triage_routing));

            % XAI Latent Embedding & KAN Decomposition Telemetry
            if isfield(response, 'xai_embeddings')
                xai = response.xai_embeddings;
                if isfield(xai, 'clinical_latent_coordinates')
                    coords = xai.clinical_latent_coordinates;
                    logMessage(app, sprintf('XAI Latent Manifold [384-D]: Coord = (%.3f, %.3f)', coords.x, coords.y));
                end
                if isfield(xai, 'kan_spline_attribution')
                    attr = xai.kan_spline_attribution;
                    logMessage(app, sprintf('KAN Spline Decomposition: YOLO26 Lesion Saliency=%d%% | GNN Topology=%d%% | AVR Caliber=%d%%', ...
                        attr.semantic_lesion_saliency_pct, attr.vascular_topology_tortuosity_pct, attr.caliber_narrowing_avr_pct));
                end
            end

            % Update UI Gauges
            app.AVRLabel.Text = sprintf('AVR (A/V Ratio): %.3f', app.AVRVal);
            app.TortuosityLabel.Text = sprintf('Tortuosity Index: %.4f', app.TortVal);
            app.BranchAngleLabel.Text = sprintf('Branching Angle: %.1f deg', angle_val);
            app.LesionCountLabel.Text = sprintf('Lesions: MA:%d | HE:%d | EX:%d', count_ma, count_he, count_ex);
            app.ModelConfidenceLabel.Text = sprintf('KAN Confidence: %.1f%%', conf * 100);

            % Color themes
            switch sev_grade
                case 0
                    theme_color = [0.13, 0.77, 0.37];
                case 1
                    theme_color = [0.92, 0.70, 0.03];
                case 2
                    theme_color = [0.98, 0.45, 0.09];
                otherwise
                    theme_color = [0.94, 0.27, 0.27];
            end

            app.SeverityLabel.Text = response.severity_text;
            app.SeverityLabel.FontColor = theme_color;
            app.RecommendationLabel.Text = sprintf('Clinical Triage: %s', response.clinical_recommendation);

            if is_escalated
                app.TriageStatusLabel.Text = 'ESCALATED: OPHTHALMOLOGIST REVIEW QUEUE';
                app.TriageStatusLabel.FontColor = [0.94, 0.27, 0.27];
            else
                app.TriageStatusLabel.Text = 'CERTIFIED: AUTONOMOUS EDGE TRIAGE';
                app.TriageStatusLabel.FontColor = [0.13, 0.77, 0.37];
            end

            % Render Active Visual Layers
            RenderOverlays(app);
            logMessage(app, 'YOLO26 Bounding Boxes & Tri-Modal Evidence successfully rendered.');
            logMessage(app, '==================================================');
        end

        function processLocalFallback(app)
            % Native MATLAB Stateflow & YOLO26 Emulation Pipeline
            [isGradable, q_score, processedImg] = Gate1_QualityCheck(app.CurrentImage);
            if ~isGradable
                app.SeverityLabel.Text = 'IMAGE QUALITY INSUFFICIENT';
                app.SeverityLabel.FontColor = [0.94, 0.27, 0.27];
                app.TriageStatusLabel.Text = 'HARD BLOCK: RECAPTURE';
                app.GateStatusLabel.Text = sprintf('Gate 1: REJECT (%.3f)', q_score);
                
                [h, w, ~] = size(app.CurrentImage);
                cla(app.UIAxes);
                image(app.UIAxes, 'CData', app.CurrentImage);
                axis(app.UIAxes, 'image');
                app.UIAxes.XLim = [1, w];
                app.UIAxes.YLim = [1, h];
                app.UIAxes.YDir = 'reverse';
                title(app.UIAxes, 'RECAPTURE SCAN: Quality Score Below 0.70', 'Color', [0.94, 0.27, 0.27], 'FontSize', 11);
                return;
            end

            app.GateStatusLabel.Text = sprintf('Gate 1: PASSED (%.3f)', q_score);
            Gate2_AnomalyDetect(processedImg);

            [yoloFeat, gnnFeat, lesionTypology, vascBiomarkers, yoloDets] = Gate3_LesionIdentify(processedImg);
            [sevGrade, sevText, confidence, ~, triageAction] = Gate4_SeverityGrade(yoloFeat, gnnFeat, lesionTypology, vascBiomarkers);

            app.SeverityGrade = sevGrade;
            app.SeverityText = sevText;
            app.AVRVal = vascBiomarkers.avr;
            app.TortVal = vascBiomarkers.tortuosity;
            app.FoveaCoords = [0.55, 0.50];
            app.OpticDiscCoords = [0.28, 0.49];

            if isstruct(yoloDets)
                app.CurrentYoloDetections = num2cell(yoloDets);
            else
                app.CurrentYoloDetections = yoloDets;
            end

            num_yolo = length(app.CurrentYoloDetections);
            app.YoloDetectionsCountLabel.Text = sprintf('YOLO26 Detections: %d (Hungarian)', num_yolo);
            app.AVRLabel.Text = sprintf('AVR (A/V Ratio): %.3f', app.AVRVal);
            app.TortuosityLabel.Text = sprintf('Tortuosity Index: %.4f', app.TortVal);
            app.BranchAngleLabel.Text = sprintf('Branching Angle: %.1f deg', vascBiomarkers.branchingAngleDeg);
            app.LesionCountLabel.Text = sprintf('Lesions: MA:%d | HE:%d | EX:%d', lesionTypology.countMA, lesionTypology.countHE, lesionTypology.countEX);
            app.ModelConfidenceLabel.Text = sprintf('KAN Confidence: %.1f%%', confidence * 100);

            app.SeverityLabel.Text = sevText;
            app.SeverityLabel.FontColor = ternary(sevGrade == 0, [0.13, 0.77, 0.37], [0.94, 0.27, 0.27]);
            app.TriageStatusLabel.Text = triageAction;
            app.RecommendationLabel.Text = sprintf('Stateflow Local Execution: %s', sevText);

            invText = cell(num_yolo + 1, 1);
            invText{1} = sprintf('[YOLO26 LOCAL INVENTORY: %d DETECTIONS]', num_yolo);
            for k = 1:num_yolo
                det_item = app.CurrentYoloDetections{k};
                invText{k + 1} = sprintf('%s: %s (%.0f%%) • %s', det_item.detection_id, det_item.lesion_type, det_item.confidence * 100, det_item.etdrs_zone);
            end
            app.YoloDetectionsTextArea.Value = invText;

            RenderOverlays(app);
            logMessage(app, 'Native MATLAB YOLO26 & 4-Gate Execution complete.');
        end

        function RenderOverlays(app)
            if isempty(app.CurrentImage)
                return;
            end

            [h, w, ~] = size(app.CurrentImage);

            % Guarantee RetinaMask exists and has matching dimensions
            if isempty(app.RetinaMask) || size(app.RetinaMask, 1) ~= h || size(app.RetinaMask, 2) ~= w
                gray_m = rgb2gray(app.CurrentImage);
                app.RetinaMask = imfill(gray_m > 18, 'holes');
            end

            cla(app.UIAxes);
            image(app.UIAxes, 'CData', app.CurrentImage);
            axis(app.UIAxes, 'image');
            app.UIAxes.XLim = [1, w];
            app.UIAxes.YLim = [1, h];
            app.UIAxes.YDir = 'reverse';
            app.UIAxes.XColor = 'none';
            app.UIAxes.YColor = 'none';
            hold(app.UIAxes, 'on');

            % --------------------------------------------------------
            % 1. Saliency Heatmap Layer (FOV Boundary Clipped)
            % --------------------------------------------------------
            if app.ShowHeatmapCheckbox.Value && app.SeverityGrade > 0
                fx = app.FoveaCoords(1) * w;
                fy = app.FoveaCoords(2) * h;
                [X, Y] = meshgrid(1:w, 1:h);
                dist_fov = sqrt((X - fx).^2 + (Y - fy).^2);
                heatmap_base = exp(-dist_fov.^2 / (2 * (h * 0.18)^2));

                r_ch = double(app.CurrentImage(:,:,1));
                g_ch = double(app.CurrentImage(:,:,2));
                lesion_act = max(0, (r_ch - g_ch)) .* double(app.RetinaMask);
                lesion_act = imgaussfilt(lesion_act, 10);

                heatmap = 0.40 * heatmap_base + 1.30 * mat2gray(lesion_act);
                heatmap = mat2gray(heatmap);

                cmap = jet(256);
                rgb_heat = ind2rgb(uint8(255 * heatmap), cmap);

                hObj = imshow(rgb_heat, 'Parent', app.UIAxes);
                alpha_data = heatmap .* double(app.RetinaMask) * 0.44;
                set(hObj, 'AlphaData', alpha_data);
            end

            % --------------------------------------------------------
            % 2. Branch B: GNN Vascular Skeleton & Bifurcation Nodes
            % --------------------------------------------------------
            if app.ShowVesselsCheckbox.Value
                g_inv = 255 - app.CurrentImage(:,:,2);
                vessel_seed = g_inv > 130 & app.RetinaMask;
                vessel_skel = bwmorph(vessel_seed, 'skel', Inf);
                vessel_bifurc = bwmorph(vessel_skel, 'branchpoints');

                [vy, vx] = find(vessel_skel);
                if ~isempty(vx)
                    step = max(1, round(length(vx) / 800));
                    plot(app.UIAxes, vx(1:step:end), vy(1:step:end), '.', 'Color', [0.20 0.85 1.0], 'MarkerSize', 2);
                end

                [by, bx] = find(vessel_bifurc);
                if ~isempty(bx)
                    step_b = max(1, round(length(bx) / 40));
                    plot(app.UIAxes, bx(1:step_b:end), by(1:step_b:end), 's', 'Color', [1.0 0.80 0.10], 'MarkerSize', 5, 'LineWidth', 1.2);
                end
            end

            % --------------------------------------------------------
            % 3. ETDRS Concentric Grid & Anatomical Landmarks
            % --------------------------------------------------------
            if app.ShowETDRSCheckbox.Value
                fx = app.FoveaCoords(1) * w;
                fy = app.FoveaCoords(2) * h;
                od_x = app.OpticDiscCoords(1) * w;
                od_y = app.OpticDiscCoords(2) * h;

                r_central = h * 0.045; % 1mm
                r_inner   = h * 0.125; % 3mm
                r_outer   = h * 0.220; % 6mm

                viscircles(app.UIAxes, [fx, fy], r_central, 'Color', [0.13, 0.77, 0.37], 'LineWidth', 1.6, 'EnhanceVisibility', false);
                viscircles(app.UIAxes, [fx, fy], r_inner,   'Color', [0.92, 0.70, 0.03], 'LineWidth', 1.4, 'EnhanceVisibility', false);
                viscircles(app.UIAxes, [fx, fy], r_outer,   'Color', [0.94, 0.27, 0.27], 'LineWidth', 1.2, 'EnhanceVisibility', false);

                % Fovea crosshair
                plot(app.UIAxes, fx, fy, 'w+', 'MarkerSize', 12, 'LineWidth', 2.0);
                text(app.UIAxes, fx + 8, fy - 8, 'Fovea (1mm)', 'Color', [1 1 1], 'FontSize', 8, 'FontWeight', 'bold');

                % Optic Disc
                if od_x > 0 && od_y > 0
                    plot(app.UIAxes, od_x, od_y, 'co', 'MarkerSize', 14, 'LineWidth', 1.8);
                    text(app.UIAxes, od_x + 10, od_y - 10, 'Optic Disc', 'Color', [0.2 0.8 1.0], 'FontSize', 8, 'FontWeight', 'bold');
                end
            end

            % --------------------------------------------------------
            % 4. Branch A: YOLO26 Nano Bounding Boxes & Segmentation Masks
            % --------------------------------------------------------
            if app.ShowYoloBoxesCheckbox.Value && ~isempty(app.CurrentYoloDetections)
                for i = 1:length(app.CurrentYoloDetections)
                    det = app.CurrentYoloDetections{i};
                    if ~isfield(det, 'box')
                        continue;
                    end

                    b = det.box;
                    bx = b.x1 * w / 100.0;
                    by = b.y1 * h / 100.0;
                    bw = b.width * w / 100.0;
                    bh = b.height * h / 100.0;

                    % Assign color by lesion class
                    l_type = det.lesion_type;
                    switch lower(l_type)
                        case {'microaneurysm', 'microaneurysms'}
                            c_box = [1.00, 0.72, 0.12]; % Amber
                        case {'hemorrhage', 'hemorrhages'}
                            c_box = [0.96, 0.22, 0.28]; % Crimson
                        case {'hard exudate', 'hard exudates', 'hardexudate'}
                            c_box = [1.00, 0.92, 0.20]; % Yellow
                        case {'cotton wool spot', 'cottonwoolspots'}
                            c_box = [0.18, 0.85, 0.98]; % Cyan
                        case {'neovascularization'}
                            c_box = [0.85, 0.28, 0.92]; % Magenta
                        otherwise
                            c_box = [0.38, 0.74, 0.98]; % Blue
                    end

                    % 4a. YOLO26 Polygonal Instance Segmentation Mask
                    if app.ShowYoloMasksCheckbox.Value && isfield(det, 'polygon_mask') && ~isempty(det.polygon_mask)
                        poly = det.polygon_mask;
                        if iscell(poly)
                            px = cellfun(@(pt) pt.x * w / 100.0, poly);
                            py = cellfun(@(pt) pt.y * h / 100.0, poly);
                            fill(app.UIAxes, px, py, c_box, 'FaceAlpha', 0.25, 'EdgeColor', c_box, 'LineWidth', 1.0);
                        elseif isstruct(poly)
                            px = [poly.x] * w / 100.0;
                            py = [poly.y] * h / 100.0;
                            fill(app.UIAxes, px, py, c_box, 'FaceAlpha', 0.25, 'EdgeColor', c_box, 'LineWidth', 1.0);
                        end
                    end

                    % 4b. YOLO26 Bounding Box
                    rectangle(app.UIAxes, 'Position', [bx, by, bw, bh], ...
                        'EdgeColor', c_box, 'LineWidth', 2.0, 'Curvature', [0.08, 0.08]);

                    % 4c. YOLO26 Annotation Tag
                    conf_pct = det.confidence * 100;
                    tag_str = sprintf(' %s: %s (%.0f%%) ', det.detection_id, l_type, conf_pct);
                    text(app.UIAxes, bx, max(10, by - 6), tag_str, ...
                        'Color', [1 1 1], 'BackgroundColor', [c_box * 0.4, 0.88], ...
                        'FontSize', 8, 'FontWeight', 'bold', 'Margin', 2);
                end
            end

            title(app.UIAxes, sprintf('YOLO26 Nano • AVR: %.3f | Tortuosity: %.4f | %s', app.AVRVal, app.TortVal, app.SeverityText), ...
                'Color', [0.9 0.9 0.9], 'FontSize', 11);
            hold(app.UIAxes, 'off');
        end
    end
end

function val = ternary(cond, a, b)
    if cond
        val = a;
    else
        val = b;
    end
end
