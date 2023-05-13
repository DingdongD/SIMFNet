clc;clear;
addpath('./utils/');  
addpath('./radar_config/'); 
addpath('./pipeline/');  

%% path configuration
root_dir = "H:\MyDataset\MatFile\";  % root path
image_dir = "H:\MyDataset\ImageFile4\";  % image save path
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
tic
for act_idx = 1 : length(act_type)
    for exp_idx = 1 : length(exp_type)
        for usr_idx = 1 : length(usr_type)
            data_dir = strcat(root_dir, act_type(act_idx), '\', exp_type(exp_idx), '\', usr_type(usr_idx), '\');
            data_files = dir(data_dir);
            for file_idx = 3 : length(data_files)
                image_folder = strcat(image_dir,act_type(act_idx), '\', exp_type(exp_idx), '\', usr_type(usr_idx), '\', data_files(file_idx).name, '\');
                disp("Being processed");
                
                % Below you can use the raw bin file for signal processing and data generation 
%                 adc_file = strcat(data_dir, data_files(file_idx).name, '\adc_data_Raw_0.bin');  
%                 [adc_data] = readDCA1000(adc_file);
%                 adc_data = adc_data(1,:);  
%                 frame_num = floor(size(adc_data,2)/(AWR1243_CONFIG.adc_sample*AWR1243_CONFIG.num_tx*AWR1243_CONFIG.chirp_num)); 
%                 adc_data = adc_data(1,1:AWR1243_CONFIG.adc_sample*AWR1243_CONFIG.num_tx*AWR1243_CONFIG.chirp_num*frame_num);
                
                % Below you can use the converted mat file for signal processing and data generation 
                adc_file = strcat(data_dir, data_files(file_idx).name, '\adc_data.mat');  
                adc_data = load(adc_file);
                adc_data = adc_data.adc_data;
                frame_num = floor(size(adc_data,2)/(AWR1243_CONFIG.adc_sample*AWR1243_CONFIG.num_tx*AWR1243_CONFIG.chirp_num));
                
                RadarPipeline(adc_data,AWR1243_CONFIG.num_tx,1,AWR1243_CONFIG.adc_sample,AWR1243_CONFIG.chirp_num,AWR1243_CONFIG.range_fft_num,...
                    AWR1243_CONFIG.doppler_fft_num,frame_num,overlap,range_of_interest,frame_interval,image_folder);
            end
       end
    end
end
toc






