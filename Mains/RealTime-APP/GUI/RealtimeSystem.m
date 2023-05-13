
%% 本文件用于实现准实时处理系统
clc;clear;close all;
data_name = "trial8";
%% 创建文件夹
root_path = 'H:\RadarProcessing\DataFile\TestData\'; % 根路径名称
data_path = strcat(root_path,data_name); 
mkdir(data_path);

TOTAL_FRAME_DURATION = 400; % 帧长为200帧
FRAME_INTERVAL = 100; % 脚本文件设置为20帧
filename = strcat(root_path,data_name,'\adc_data_Raw_0.bin'); % 检测该文件夹下是否存在bin文件
FRAME_AXIS = (1:TOTAL_FRAME_DURATION)*100/1e3; % 帧时间
file_path = strcat(root_path,data_name,'\');

RT_SET = [];
DT_SET = [];

figure(1);
tic 
for frame_index = 1:TOTAL_FRAME_DURATION/FRAME_INTERVAL
    % 发送指令
    SendCaptureCMD(data_name);
    path_size = 0; % 初始化bin文件大小
    % 读取并处理文件
    while(path_size ~= 76800*2) % 如果该文件夹不存在则创建一个该目录
       D = dir(filename);
       path_size = D.bytes / 1024; % 文件大小
       disp('文件正在采集');
    end
    [Range_Time_Plot,RANGE_AXIS,Micro_Doppler_Plot,VELOCITY_AXIS] = ProcessData(filename);
    RT_SET = cat(2,RT_SET,squeeze(Range_Time_Plot));
    DT_SET = cat(2,DT_SET,squeeze(Micro_Doppler_Plot));
    FRAME_SET = FRAME_AXIS(1+FRAME_INTERVAL*(frame_index-1):FRAME_INTERVAL*frame_index);
    subplot(121);
    imagesc(FRAME_SET,RANGE_AXIS,db(squeeze((RT_SET)))); % RANGE_AXIS,
    xlabel('Frame Period(s)');ylabel('Range(m)');
    subplot(122);
    imagesc(FRAME_SET,VELOCITY_AXIS,db(squeeze(DT_SET)));
    xlabel('Frame Period(s)');ylabel('Velocity(m/s)');
    pause(0.01)
    change_name = strcat(file_path,'adc_data_Raw_0',num2str(frame_index),'.bin');
    movefile(filename,change_name);
end
toc


