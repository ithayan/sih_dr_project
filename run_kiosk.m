function appOut = run_kiosk(mode)
% =========================================================================
% OCUNEXA CLINICAL AI WORKSTATION LAUNCHER (SIH26038 MathWorks Track)
% =========================================================================
% 1-Click launcher for the OcuNexa Clinical Telemedicine Workstation.
% Architecture: Hybrid YOLO26-GNN-KAN with Stateflow Hard Quality Gates.
% Backend: http://127.0.0.1:8080 (server.py)
%
% Usage:
%   app = run_kiosk();           % Launches React.js Kiosk inside MATLAB desktop (uihtml)
%   run_kiosk('browser');        % Launches directly in System Browser (Fastest, zero CSP)
%   app = run_kiosk('classic');  % Launches classic MATLAB App Designer visualizer

    if nargin < 1 || isempty(mode)
        mode = 'react';
    end

    disp('========================================================================');
    disp('  OCUNEXA: CLINICAL AI SCREENING & TELEMEDICINE KIOSK LAUNCHER');
    disp('  Backend Endpoint: http://127.0.0.1:8080');
    disp('========================================================================');

    switch lower(mode)
        case 'browser'
            disp('Opening OcuNexa Kiosk in default system browser...');
            web('http://127.0.0.1:8080', '-browser');
            appInstance = [];

        case 'classic'
            disp('Launching Classic MATLAB App Designer Visualizer...');
            appInstance = XAI_Visualizer();

        case {'react', 'matlab', 'uihtml'}
            disp('Launching Embedded React.js Kiosk inside MATLAB desktop window...');
            appInstance = XAI_React_Kiosk(8080);

        otherwise
            warning('Unrecognized mode "%s". Defaulting to embedded React kiosk.', mode);
            appInstance = XAI_React_Kiosk(8080);
    end

    if nargout > 0
        appOut = appInstance;
    end
end
