% run_demo.m
% Script to execute a test run of the pipeline

% Test Case 1: Standard Image
disp('==================== TEST CASE 1 ====================');
[result1, exp1] = Main_Pipeline('patient_001_left_eye.jpg');

% Pause for readability
pause(2);

% Test Case 2: Another Image
disp('==================== TEST CASE 2 ====================');
[result2, exp2] = Main_Pipeline('patient_002_right_eye.jpg');
