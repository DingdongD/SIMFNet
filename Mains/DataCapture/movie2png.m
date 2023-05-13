clc;clear;close all;
%% read mp4
video_file='D:\MyDataset\MovieFile\2022-07-21_13-32-58.mp4';
video=VideoReader(video_file);
frame_number=floor(video.Duration * video.FrameRate);
file_name = 'H:\RadarProcessing\videotest\subject1\';

%% convert mp4 to png
for i=[96:2:120]
    image_name=strcat(file_name,'distress',num2str(i));
    image_name=strcat(image_name,'.png');
    I=read(video,i);                             
    imshow(I)
    imwrite(I,image_name,'png');               
    I=[];
end

% delete('*.png')