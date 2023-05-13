clc;clear;
% By Xuliang,22134033@zju.edu.cn
% FUNC: convert bin file to mat file

addpath('./utils/');  
addpath('./radar_config/'); 
addpath('./pipeline/');  

root_dir = "H:\MyDataset\DatasetFile\";  % root path
mat_dir = "H:\MyDataset\MatFile\";  % image save path
act_type = ["drown","distress","freestyle","backstroke","breaststroke","float_with_a_ring",...
    "pull_with_a_ring","swim_with_a_ring","frolic","wave"];  % activity class
exp_type = ["radial_shallow","non_radial_shallow","radial_deep"];  % scene class
usr_type = ["user_1","user_2"];  % user class

%% load radar parameter configure files
load('./radar_config/AWR1243_CONFIG.mat'); 
range_of_interest = 20:120; % ROI
overlap = 0.5; % overlapping rate
frame_interval = [1,40]; % Sliding window length for intercepting maps

%% To generate dataset
% for act_idx = 7 : 7
for act_idx = 1 : length(act_type)
    for exp_idx = 1 : length(exp_type)
        for usr_idx = 1 : length(usr_type)
            data_dir = strcat(root_dir, act_type(act_idx), '\', exp_type(exp_idx), '\', usr_type(usr_idx), '\');
            data_files = dir(data_dir);
            for file_idx = 3 : length(data_files)
                adc_file = strcat(data_dir, data_files(file_idx).name, '\adc_data_Raw_0.bin'); 
                mat_dir_path = strcat(mat_dir,act_type(act_idx), '\', exp_type(exp_idx), '\', usr_type(usr_idx), '\', data_files(file_idx).name, '\');
                mkdir(mat_dir_path);
                mat_file = strcat(mat_dir_path,'adc_data.mat');
                disp("Being Processing");
                [adc_data] = readDCA1000(adc_file);
%                 adc_data = adc_data(1,:);  % Select one RX antenna data
                frame_num = floor(size(adc_data,2)/(AWR1243_CONFIG.adc_sample*AWR1243_CONFIG.num_tx*AWR1243_CONFIG.chirp_num));
                adc_data = adc_data(1,1:AWR1243_CONFIG.adc_sample*AWR1243_CONFIG.num_tx*AWR1243_CONFIG.chirp_num*frame_num);
                
                save(mat_file,'adc_data');
            end
       end
    end
end







