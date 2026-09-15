% build_webapp.m
% =========================================================================
% NetraX: MATLAB Web App Archive (.ctf) Compiler Script (SIH26038)
% Packages XAI_Visualizer into a deployable MATLAB Web App (.ctf)
% for hosting on MATLAB Web App Server (Port 31415)
% =========================================================================

disp('========================================================================');
disp('  NETRAX: COMPILING MATLAB WEB APP ARCHIVE (.ctf)');
disp('  Target: MATLAB Web App Server');
disp('========================================================================');

outDir = fullfile(pwd, 'dist_webapp');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Build Web App Archive (.ctf)
% Includes all Stateflow gates, sample assets, and XAI Visualizer
try
    disp('Executing MATLAB Compiler (mcc)...');
    mcc('-W', 'webapp:NetraX_Clinical_Workstation', ...
        '-d', outDir, ...
        'XAI_Visualizer.m', ...
        '-a', 'Gate1_QualityCheck.m', ...
        '-a', 'Gate2_AnomalyDetect.m', ...
        '-a', 'Gate3_LesionIdentify.m', ...
        '-a', 'Gate4_SeverityGrade.m', ...
        '-a', 'test_fundus_valid.png', ...
        '-a', 'sample_grade0_normal.png', ...
        '-v');
    
    disp('========================================================================');
    disp(['SUCCESS: Web App Archive generated in: ', fullfile(outDir, 'NetraX_Clinical_Workstation.ctf')]);
    disp('To deploy:');
    disp('1. Copy NetraX_Clinical_Workstation.ctf to your MATLAB Web App Server apps folder.');
    disp('2. Access via browser: http://localhost:31415/webapps/home/');
    disp('3. Expose to judges via tunnel: ngrok http 31415');
    disp('========================================================================');
catch ME
    disp(['COMPILATION NOTICE: ', ME.message]);
    disp('Ensure MATLAB Compiler toolbox is installed and licensed.');
end
