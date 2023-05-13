%% To classify quatic human activity using SVM
%% By Xuliang,22134033@zju.edu.cn

clc;clear;close all;
train_data = load('train_dataset.mat');
test_data = load('test_dataset.mat');
train_data = train_data.train_set;
test_data = test_data.test_set;

enable_train_data_idx = train_data(:,end)==1 | train_data(:,end)==2 | train_data(:,end)==3 |train_data(:,end)==4;
disable_train_data_idx = ~enable_train_data_idx;
enable_test_data_idx = test_data(:,end)==1 | test_data(:,end)==2 | test_data(:,end)==3 |test_data(:,end)==4;
disable_test_data_idx = ~enable_test_data_idx;

enable_train_data = train_data(enable_train_data_idx,:);
enable_train_data(:,end)=-1; 
disable_train_data = train_data(disable_train_data_idx,:);
disable_train_data(:,end)=1;
enable_test_data = test_data(enable_test_data_idx,:);
enable_test_data(:,end)=-1;
disable_test_data = test_data(disable_test_data_idx,:);
disable_test_data(:,end)=1;

train_data = [enable_train_data(:,1:end-1);disable_train_data(:,1:end-1)];
train_label = [enable_train_data(:,end);disable_train_data(:,end)];
rand_idx = randperm(length(train_data));
train_x = train_data(rand_idx,:);
train_y = train_label(rand_idx,:);

test_data = [enable_test_data(:,1:end-1);disable_test_data(:,1:end-1)];
test_label = [enable_test_data(:,end);disable_test_data(:,end)];
rand_idx = randperm(length(test_data));
test_x = test_data(rand_idx,:);
test_y = test_label(rand_idx,:);

feature_num = [3,4,7,9,10,13,18,24,25,27,28];
train_x = train_x(:,feature_num);
test_x = test_x(:,feature_num);

%% K-fold cross-test (penalty factor and RBF parameters)
[c,g] = meshgrid(-10:2:10,-10:2:10);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10e-4;
v = 5; % fold number
bestc = 1;
bestg = 0.1;
bestacc = 0;
% cmd : -v Number of cross-checks (n-fold validation mode) 
% -t denotes function type,0 denotes linear u'*v,1 denotes polynomial (gamma*u'*v + coef0)^degree£¬
% 2 denotes RBF£ºexp(-gamma*|u-v|^2)£¬3 denotes sigmoid£ºtanh(gamma*u'*v + coef0)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)

for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),'-t2','-c ',num2str(2^c(i,j)),'-g ',num2str(2^g(i,j ))];
        cg(i,j) = svmtrain(train_y,train_x,cmd);
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2 ^g(i,j);
        end
        if abs(cg(i,j)-bestacc)<=eps && bestc>2^c(i,j)
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2 ^g(i,j);
        end
    end
end

%% Extraction of different features to achieve training 
% feature_num = [1,5,11,13,16:20,23,24];

%% Select the best penalty factor and gamma parameters
cmd = ['-t 2','-c',num2str(bestc),'-g',num2str(bestg)];
model = svmtrain(train_y,train_x,cmd);
[predict_train,accuracy_1,prob_estimates] = svmpredict(train_y,train_x,model);
train_kind_num = tabulate(train_y);train_kind_num = train_kind_num(:,2);
accuracy = sum(predict_train==train_y)/sum(train_kind_num);

figure(1);
is_normalise = false;
[train_f1,train_acc,train_confusion_matrix] = ConfusionMat(train_y,predict_train);
Plot_Confusion_Map({'Drowning people','Normal people'},train_confusion_matrix)
% confusion_matrix_plot(train_confusion_matrix,is_normalise);

figure(2);
[predict_test,accuracy_2,prob_estimates2] = svmpredict(test_y,test_x,model);
[test_f1,test_acc,test_confusion_matrix] = ConfusionMat(test_y,predict_test);
Plot_Confusion_Map({'Drowning people','Normal people'},test_confusion_matrix)

