%% By Xuliang,22134033@zju.edu.cn
%% FUNC:To construct the processed radar image dataset

clc;clear;close all;
root_path = 'H:\RadarProcessing\Data\Images3\';
act = 'pull_buoy';

if strcmp(act,'struggle')
    name_id = 'A';
elseif strcmp(act,'wave_buoy')
    name_id = 'B';
elseif strcmp(act,'float_buoy')
    name_id = 'C';
elseif strcmp(act,'float')
    name_id = 'D';
elseif strcmp(act,'freestyle')
    name_id = 'E';
elseif strcmp(act,'breaststroke')
    name_id = 'F';
elseif strcmp(act,'backstroke')
    name_id = 'G';
elseif strcmp(act,'swim_buoy')
    name_id = 'H';
elseif strcmp(act,'pull_buoy')
    name_id = 'I';
end

md_file_path = strcat(root_path,act,'\md\');
rt_file_path = strcat(root_path,act,'\rt\');
cvd_file_path = strcat(root_path,act,'\cvd\');
% svm_file_path = strcat(root_path,act,'\feature\');
md_files = dir(md_file_path);
rt_files = dir(rt_file_path);
cvd_files = dir(cvd_file_path);
% svm_files = dir(svm_file_path);

md_oldName = cell(length(md_files)-2,1); 
rt_oldName = cell(length(rt_files)-2,1); 
cvd_oldName = cell(length(cvd_files)-2,1); 
% svm_oldName = cell(length(svm_files)-2,1); 

md_newName = cell(length(md_files)-2,1); 
rt_newNme = cell(length(rt_files)-2,1); 
cvd_newName = cell(length(cvd_files)-2,1); 
% svm_newName = cell(length(svm_files)-2,1); 

for i = 3:length(md_files)
    md_oldName{i-2} = md_files(i).name;
end
for i = 3:length(rt_files)
    rt_oldName{i-2} = rt_files(i).name;
end
for i = 3:length(rt_files)
    cvd_oldName{i-2} = cvd_files(i).name;
end
% for i = 3:length(rt_files)
%     svm_oldName{i-2} = svm_files(i).name;
% end

data_length = length(rt_oldName);
data_idx = randperm(data_length);
train_num = round(data_length * 1);
train_idx = data_idx(1:train_num);
% test_idx = data_idx(train_num+1:end);

new_file = 'H:\RadarProcessing\Data\Images3\Data_v4\';
for j = 1:length(train_idx)
   md_newName{j} = strcat(name_id,'MD_',num2str(j),'.jpg') ; 
   copyfile([md_file_path md_oldName{j}], [new_file 'train_md\'  md_newName{j}]) 
   
   rt_newName{j} = strcat(name_id,'RT_',num2str(j),'.jpg') ; 
   copyfile([rt_file_path rt_oldName{j}], [new_file 'train_rt\' rt_newName{j}]) 
   
   cvd_newName{j} = strcat(name_id,'CVD_',num2str(j),'.jpg') ; 
   copyfile([cvd_file_path cvd_oldName{j}], [new_file 'train_cvd\' cvd_newName{j}]) 
   
   svm_newName{j} = strcat(name_id,'SVM_',num2str(j),'.mat') ; 
   copyfile([svm_file_path svm_oldName{j}], [new_file 'train_svm\' svm_newName{j}]) 

%    md_newName{j} = strcat(name_id,'MD_',num2str(j),'.jpg') ; 
%    copyfile([md_file_path md_oldName{j}], [new_file 'test_md\'  md_newName{j}]) 
%    
%    rt_newName{j} = strcat(name_id,'RT_',num2str(j),'.jpg') ; 
%    copyfile([rt_file_path rt_oldName{j}], [new_file 'test_rt\' rt_newName{j}]) 
%    
%    cvd_newName{j} = strcat(name_id,'CVD_',num2str(j),'.jpg') ; 
%    copyfile([cvd_file_path cvd_oldName{j}], [new_file 'test_cvd\' cvd_newName{j}]) 

%    svm_newName{j} = strcat(name_id,'SVM_',num2str(j),'.mat') ; 
%    copyfile([svm_file_path svm_oldName{j}], [new_file 'test_svm\' svm_newName{j}]) 
end


