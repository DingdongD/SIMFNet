% 本文件用于构建域自适应的数据集
clc;clear;close all;

root_dir = "H:\MyDataset\ImageFile\";  % 根路径
image_dir = "H:\MyDataset\DomainSet\";  % 新数据保存路径
act_type = ["distress","drown","freestyle","breaststroke","backstroke","pull_with_a_ring",...
    "swim_with_a_ring","float_with_a_ring","wave","frolic"];  % 行为类型
exp_type = ["radial_shallow","non_radial_shallow","radial_deep"];  % 场景类型
usr_type = ["user_1","user_2"];  % 用户类型
data_type = ["TR","TD","CVD"];
label_type = ['A','B','C','D','E','F','G','H','I','J'];

for act_idx = 1 : length(act_type)
    for exp_idx = 1 : length(exp_type)
        for usr_idx = 1 : length(usr_type)
            data_dir = strcat(root_dir, act_type(act_idx), '\', exp_type(exp_idx), '\', usr_type(usr_idx), '\');
            data_files = dir(data_dir);
            for file_idx = 3 : length(data_files)
                old_data_dir = strcat(data_dir, data_files(file_idx).name, '\');
                old_data_files = dir(old_data_dir);  
                for old_file_idx = 1:length(data_type)
                    old_image_dir = strcat(old_data_dir,data_type(old_file_idx),'\');
                    old_image_files = dir(old_image_dir);
                    
                    for old_image_idx = 3 : length(old_image_files)
                        tocopy_file = strcat(old_image_dir,old_image_files(old_image_idx).name);
                        if exp_idx == 1 && usr_idx == 1
                            new_file_name1 = strcat(image_dir,'Domain_1','\',data_type(old_file_idx),'\',label_type(act_idx),'_',data_files(file_idx).name,'_',old_image_files(old_image_idx).name);
                            copyfile(tocopy_file, new_file_name1);
                        elseif exp_idx == 1 && usr_idx == 2
                            new_file_name2 = strcat(image_dir,'Domain_2','\',data_type(old_file_idx),'\',label_type(act_idx),'_',data_files(file_idx).name,'_',old_image_files(old_image_idx).name);
                            copyfile(tocopy_file, new_file_name2);
                        elseif exp_idx == 2 && usr_idx == 1
                            new_file_name3 = strcat(image_dir,'Domain_3','\',data_type(old_file_idx),'\',label_type(act_idx),'_',data_files(file_idx).name,'_',old_image_files(old_image_idx).name);
                            copyfile(tocopy_file, new_file_name3);
                        elseif exp_idx == 2 && usr_idx == 2
                            new_file_name4 = strcat(image_dir,'Domain_4','\',data_type(old_file_idx),'\',label_type(act_idx),'_',data_files(file_idx).name,'_',old_image_files(old_image_idx).name);
                            copyfile(tocopy_file, new_file_name4);
                        elseif exp_idx == 3 && usr_idx == 1
                            new_file_name5 = strcat(image_dir,'Domain_5','\',data_type(old_file_idx),'\',label_type(act_idx),'_',data_files(file_idx).name,'_',old_image_files(old_image_idx).name);
                            copyfile(tocopy_file, new_file_name5);
                        elseif exp_idx == 3 && usr_idx == 2
                            new_file_name6 = strcat(image_dir,'Domain_6','\',data_type(old_file_idx),'\',label_type(act_idx),'_',data_files(file_idx).name,'_',old_image_files(old_image_idx).name);
                            copyfile(tocopy_file, new_file_name6);
                        end   
                    end
                end
            end 
        end
    end
end



