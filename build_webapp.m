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
try
    disp('Executing MATLAB Compiler (mcc with -C for App Designer UI)...');
    mcc('-W', 'webapp:NetraX_Clinical_Workstation', ...
        '-C', ...
        '-d', outDir, ...
        'launch_NetraX_Web.m', ...
        '-a', 'XAI_Visualizer.m', ...
        '-a', 'Gate1_QualityCheck.m', ...
        '-a', 'Gate2_AnomalyDetect.m', ...
        '-a', 'Gate3_LesionIdentify.m', ...
        '-a', 'Gate4_SeverityGrade.m', ...
        '-a', 'test_fundus_valid.png', ...
        '-a', 'sample_grade0_normal.png', ...
        '-v');
    
    disp('========================================================================');
    ctfFile = fullfile(outDir, 'NetraX_Clinical_Workstation.ctf');
    if exist(ctfFile, 'file')
        disp(['SUCCESS: Web App Archive generated in: ', ctfFile]);
        s = dir(ctfFile);
        disp(['Archive Size: ', num2str(s.bytes / 1024, '%.1f'), ' KB']);
    else
        disp(['Build finished. Output directory: ', outDir]);
        dir(outDir);
    end
    disp('========================================================================');
catch ME
    disp(['COMPILATION ERROR: ', ME.message]);
end
