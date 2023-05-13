clear all; clc
%% 本文件用于实现控制摄像头录像
tic
closepreview;
video_information = imaqhwinfo();
% video_format = video_information.DeviceInfo.SupportedFormats; % 可用于查看格式
% winvideoinfo = imaqhwinfo('winvideo')
vid = videoinput('winvideo',2,'RGB32_1280x720');
preview(vid)
filename = 'E:\RadarProcessing\film';
nframe = 160; % 有几帧数据  
nrate = 20; % 每秒帧率表示每秒几帧，nframe*nrate表示3s
MakeVideo(vid, filename, nframe, nrate, 2) % 开始生成图像数据
toc