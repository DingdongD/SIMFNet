clc;clear;close all;
%% By Xuliang,22134033@zju.edu.cn
%% FUNC: An SVM detector that implements a false alarm adjustable for identifying drowning and safety personnel

train_data = load('train_v3.mat');
test_data = load('test_v3.mat');

train_data = train_data.train_set;
test_data = test_data.test_set;

enable_train_data_idx = train_data(:,end)==1 | train_data(:,end)==2 | train_data(:,end)==3 |train_data(:,end)==4;
disable_train_data_idx = ~enable_train_data_idx;
enable_test_data_idx = test_data(:,end)==1 | test_data(:,end)==2 | test_data(:,end)==3 |test_data(:,end)==4;
disable_test_data_idx = ~enable_test_data_idx;

enable_train_data = train_data(enable_train_data_idx,:);
enable_train_data(:,end)=-1; % people who are unable to help themselves
disable_train_data = train_data(disable_train_data_idx,:);
disable_train_data(:,end)=1; % people who are able to help themselves
enable_test_data = test_data(enable_test_data_idx);
enable_test_data(:,end)=-1;
disable_test_data = test_data(disable_test_data_idx);
disable_test_data(:,end)=1;

PFA = 1; % real frequency
Pfa = 1e-3; % expected frequency 
min_error = 1e-4; % allowable tolerance
gamma = 0.5; % Penalty function parameters (inverse of the default category)
monte_num = 1000; % mentecarlo numbers
beta0 = [1e-3:1e-3:1];
beta1 = [2e-2:1e-2:1e-1,1e-1:1e-1:3e-1];
pfa_list = [];
pd_list = [];

feature_num = [1:18,24:28];
train_data = [enable_train_data(:,1:end-1);disable_train_data(:,1:end-1)];
train_label = [enable_train_data(:,end);disable_train_data(:,end)];
rand_idx = randperm(length(train_data));
train_data = train_data(rand_idx,feature_num );
train_label = train_label(rand_idx,:);

%% mente-carlo simulation
for k = 1 :length(beta1)
    for j = 1 : length(beta0)
        PFA_MAT = [];
        PD_MAT = [];
        for  i = 1 : monte_num
            cmd = ['-s 0',' -t2',' -c2',' -w-1 ',num2str(beta0(j)),' -w1 ',num2str(beta1(k)),' -g ',num2str(gamma)];
        %     cmd = ['-t0',' -c ',num2str(100),' -w1 ',num2str(beta_1),' -w-1 ',num2str(beta_0(i))];
            train_model = svmtrain(train_label,train_data,cmd);
            [predict_y] = svmpredict(train_label,train_data,train_model);
            PFA = sum((predict_y==1) & (train_label == -1)) / sum(train_label == -1);
            PD = sum(((predict_y==1) & (train_label == 1)) | ((predict_y==-1) & (train_label == -1))) / size(train_label,1);
            PFA_MAT = [PFA_MAT;PFA];
            PD_MAT = [PD_MAT;PD];
        end
        pfa_list(j,k) = mean(PFA_MAT);
        pd_list(j,k) = mean(PD_MAT);
    end
end

%% PFA-¦Â curve visulization
figure(1)
semilogx(beta0,(pfa_list(:,1)),'s','LineWidth',2,'Color','#B22222','MarkerIndices',1:1:length(pfa_list));hold on;
semilogx(beta0,(pfa_list(:,2)),'-d','LineWidth',2,'Color','#FA8072','MarkerIndices',1:2:length(pfa_list));hold on;
semilogx(beta0,(pfa_list(:,3)),'o','LineWidth',2,'Color','#FFD700','MarkerIndices',1:3:length(pfa_list));hold on;
semilogx(beta0,(pfa_list(:,4)),'-+','LineWidth',2,'Color','#98FB98','MarkerIndices',1:4:length(pfa_list));hold on;
semilogx(beta0,(pfa_list(:,5)),'*','LineWidth',2,'Color','#00FA9A','MarkerIndices',1:5:length(pfa_list));hold on;
semilogx(beta0,(pfa_list(:,6)),'-p','LineWidth',2,'Color','#FF4500','MarkerIndices',1:6:length(pfa_list));hold on;
semilogx(beta0,(pfa_list(:,7)),'^','LineWidth',2,'Color','#FA8072','MarkerIndices',1:7:length(pfa_list));hold on;
semilogx(beta0,(pfa_list(:,8)),'-v','LineWidth',2,'Color','#000080','MarkerIndices',1:8:length(pfa_list));hold on;
semilogx(beta0,(pfa_list(:,9)),'>','LineWidth',2,'Color','#8A2BE2','MarkerIndices',1:9:length(pfa_list));hold on;
semilogx(beta0,(pfa_list(:,11)),'-<','LineWidth',2,'Color','#C71585','MarkerIndices',1:10:length(pfa_list));hold on;

legend('\beta_{1}=0.02','\beta_{1}=0.03','\beta_{1}=0.04','\beta_{1}=0.05','\beta_{1}=0.06','\beta_{1}=0.07','\beta_{1}=0.08','\beta_{1}=0.09','\beta_{1}=0.1','\beta_{1}=0.2');grid minor;
xlabel('\beta_{0}');ylabel('PFA');

%% PD - PFA curve visulization
figure(2);
semilogy(pfa_list(:,1),pd_list(:,1),'s','LineWidth',2,'Color','#B22222','MarkerIndices',1:1:length(pfa_list));hold on;
semilogy(pfa_list(:,2),pd_list(:,2),'d','LineWidth',2,'Color','#FA8072','MarkerIndices',1:2:length(pfa_list));hold on;
semilogy(pfa_list(:,3),pd_list(:,3),'o','LineWidth',2,'Color','#FFD700','MarkerIndices',1:3:length(pfa_list));hold on;
semilogy(pfa_list(:,4),pd_list(:,4),'+','LineWidth',2,'Color','#98FB98','MarkerIndices',1:4:length(pfa_list));hold on;
semilogy(pfa_list(:,5),pd_list(:,5),'*','LineWidth',2,'Color','#00FA9A','MarkerIndices',1:5:length(pfa_list));hold on;
semilogy(pfa_list(:,6),pd_list(:,6),'p','LineWidth',2,'Color','#FF4500','MarkerIndices',1:6:length(pfa_list));hold on;
semilogy(pfa_list(:,7),pd_list(:,7),'^','LineWidth',2,'Color','#FA8072','MarkerIndices',1:7:length(pfa_list));hold on;
semilogy(pfa_list(:,8),pd_list(:,8),'v','LineWidth',2,'Color','#000080','MarkerIndices',1:8:length(pfa_list));hold on;
semilogy(pfa_list(:,9),pd_list(:,9),'>','LineWidth',2,'Color','#8A2BE2','MarkerIndices',1:9:length(pfa_list));hold on;
semilogy(pfa_list(:,11),pd_list(:,11),'<','LineWidth',2,'Color','#C71585','MarkerIndices',1:10:length(pfa_list));hold on;
xlabel('P_{fa}');ylabel('Pd');
legend('\beta_{1}=0.02','\beta_{1}=0.03','\beta_{1}=0.04','\beta_{1}=0.05','\beta_{1}=0.06','\beta_{1}=0.07','\beta_{1}=0.08','\beta_{1}=0.09','\beta_{1}=0.1','\beta_{1}=0.2');grid minor;
xlim([0 0.3])
ylim([0 0.9])

save('pfa.mat','pfa_list');
save('pd.mat','pd_list');

