classdef XAI_React_Kiosk < handle
    % XAI_React_Kiosk: Embeds the React.js OcuNexa Clinical Kiosk directly inside MATLAB
    % Uses MATLAB's modern uihtml Chromium engine for hospital-grade aesthetics.
    % Provides two-way communication between MATLAB Stateflow and the React frontend.

    properties (Access = public)
        UIFigure matlab.ui.Figure
        HTMLComponent matlab.ui.control.HTML
        ToolbarPanel matlab.ui.container.Panel
        BrowserBtn matlab.ui.control.Button
        ReloadBtn matlab.ui.control.Button
        StatusLabel matlab.ui.control.Label
        ServerURL char = 'http://127.0.0.1:8080'
    end

    methods (Access = public)
        function app = XAI_React_Kiosk(port)
            if nargin >= 1 && ~isempty(port)
                app.ServerURL = sprintf('http://127.0.0.1:%d', port);
            end

            disp('========================================================================');
            disp('  OCUNEXA: LAUNCHING REACT.JS CLINICAL KIOSK INSIDE MATLAB (uihtml)');
            disp('  Engine: MATLAB R2026a Chromium Embedded Framework (CEF)');
            disp('  Frontend: React 19 + TypeScript + TailwindCSS');
            disp(['  Backend: Hybrid YOLO-GNN-KAN Core (YOLO26 Nano + GAT + KAN) (' app.ServerURL ')']);
            disp('========================================================================');

            createComponents(app);
        end

        function reload(app)
            loaderFile = fullfile(fileparts(mfilename('fullpath')), 'kiosk_loader.html');
            if ~isfile(loaderFile)
                loaderFile = 'kiosk_loader.html';
            end
            app.HTMLComponent.HTMLSource = '';
            drawnow;
            app.HTMLComponent.HTMLSource = loaderFile;
        end

        function openInBrowser(app)
            % Bypasses any MATLAB CEF internal CSP restrictions with 100% native Chrome/Edge GPU acceleration
            web(app.ServerURL, '-browser');
        end
    end

    methods (Access = private)
        function createComponents(app)
            % 1. Create Dark-Mode Window
            app.UIFigure = uifigure(...
                'Name', 'OcuNexa • Clinical AI Workstation (React.js in MATLAB App Designer)', ...
                'Color', [0.06 0.08 0.11], ...
                'AutoResizeChildren', 'off', ...
                'Position', [50 50 1360 860]);

            % 2. Top Quick-Action Navigation Bar (For easy browser fallback if CEF triggers CSP)
            app.ToolbarPanel = uipanel(app.UIFigure, ...
                'BackgroundColor', [0.09 0.13 0.19], ...
                'BorderType', 'none', ...
                'Position', [0 820 1360 40]);

            app.StatusLabel = uilabel(app.ToolbarPanel, ...
                'Text', ['  OcuNexa Telemedicine Core: ' app.ServerURL '  |  Stateflow Quality Gates: Active'], ...
                'FontColor', [0.58 0.64 0.72], ...
                'FontSize', 12, ...
                'FontWeight', 'bold', ...
                'Position', [10 5 600 30]);

            app.BrowserBtn = uibutton(app.ToolbarPanel, 'push', ...
                'Text', 'Open in System Browser (Full GPU / No CSP)', ...
                'BackgroundColor', [0.05 0.65 0.85], ...
                'FontColor', [1 1 1], ...
                'FontWeight', 'bold', ...
                'Position', [920 6 270 28], ...
                'ButtonPushedFcn', @(src, event) app.openInBrowser());

            app.ReloadBtn = uibutton(app.ToolbarPanel, 'push', ...
                'Text', 'Reload CEF', ...
                'BackgroundColor', [0.15 0.20 0.28], ...
                'FontColor', [0.85 0.90 0.95], ...
                'Position', [1200 6 120 28], ...
                'ButtonPushedFcn', @(src, event) app.reload());

            % 3. Embed React.js App via uihtml loading local HTML bridge
            loaderFile = fullfile(fileparts(mfilename('fullpath')), 'kiosk_loader.html');
            if ~isfile(loaderFile)
                loaderFile = 'kiosk_loader.html';
            end

            app.HTMLComponent = uihtml(app.UIFigure, ...
                'HTMLSource', loaderFile, ...
                'Position', [0 0 1360 820]);

            % 4. Handle Dynamic Window Resizing
            app.UIFigure.SizeChangedFcn = @(src, event) resizeApp(app);

            % 5. Bidirectional Event Listener
            app.HTMLComponent.HTMLEventReceivedFcn = @(src, event) handleReactEvent(app, event);
        end

        function resizeApp(app)
            figPos = app.UIFigure.Position;
            w = figPos(3);
            h = figPos(4);
            if ~isempty(app.ToolbarPanel) && isvalid(app.ToolbarPanel)
                app.ToolbarPanel.Position = [0 max(0, h - 40) w 40];
                app.BrowserBtn.Position = [max(10, w - 420) 6 270 28];
                app.ReloadBtn.Position = [max(290, w - 140) 6 120 28];
            end
            if ~isempty(app.HTMLComponent) && isvalid(app.HTMLComponent)
                app.HTMLComponent.Position = [0 0 w max(10, h - 40)];
            end
        end

        function handleReactEvent(app, event)
            % Handles events dispatched from React via window.htmlComponent.sendEventToMATLAB(...)
            disp(['[React -> MATLAB Event]: ' event.EventName]);
            if isfield(event.Data, 'action')
                disp(['Action: ' event.Data.action]);
            end
        end
    end
end
