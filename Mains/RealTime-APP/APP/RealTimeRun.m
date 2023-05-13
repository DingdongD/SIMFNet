%% 主文件

clear
close all
% 定义配置
adcnum=256;
chirpnum=255;
framebytes=adcnum*chirpnum*3*4*2*2;
% 如 256adc*255chirp*3TX*4RX*2IQ*2 = 3133440
res_r=0.0499;
res_v=0.0278;
res_t=0.04;%实为0.08，STFT 50%后变为0.04

global saveName
saveName='1';%文件名

global framevalue
framevalue=100;

if exist(['D:\RadarProcessing\RawData\',saveName,'_Raw_0.bin'],'file')
	error(['文件【',saveName,'】已存在，请重命名或删除原文件']);
end

range_min=1;%1:256
range_max=128;%1:256

Mdl=load('TrainedModel.mat').oldMdl;%加载分类模型，此模型是以78个特征为基础的SVM分类模型

vf=[];
rf=[];
PLOT_ON=1;
USER_ON=1;
ANSYS_ON=0;

h1=figure(1);
h1.Position=[340 406 560 420];

%% 注重用户界面的展示 亮灯
if USER_ON
    h2=figure(2);
    h2.Position=[920 406 560 420];
    run('plottest.m');
end
%%

k=1;
oldframe=0;
oldframe_stft=0;

% MMWAVE
run('RSTD_Interface_Example.m');

starttime=clock;
tic
disp(['STARTTIME: ',num2str(starttime(4)),':',num2str(starttime(5)),':',num2str(starttime(6))]);


% 保存文件的名字
fileName=['D:\RadarProcessing\RawData\',saveName,'_Raw_0.bin'];
% 检测是否生成
while 1
    if size(dir(fileName),1)~=0
        break;
    end
end
outerbytes=0;
conframe=0;

raw_idx=1;
file=zeros(100,5);
timetable=strings(100,1);
vf_ansys=[];vf_old=[];vf_con=[];
rf_ansys=[];rf_old=[];rf_con=[];

label=9;
toc
fid = fopen(fileName,'r');

while toc<framevalue*res_t*2+5 % 检测时长，应大于采集时长5s
% read size of data real-time (2-7ms) 
    filebytes=dir(fileName).bytes-outerbytes;
    fileframe=filebytes/(framebytes); 
    newframe=floor(fileframe)+conframe;

    if newframe>oldframe+1         
        file(k,1)=toc;
        file(k,2)=newframe;
        file(k,3)=newframe-oldframe;
        
        adcData=fread(fid, (newframe-oldframe)*(framebytes/2), 'int16'); %按顺序读取
        adcData = reshape(adcData, 4*2, []);
        adcData = adcData([1,2,3,4],:) + sqrt(-1)*adcData([5,6,7,8],:);
        
        RX1_DATA = reshape(adcData(1,:),adcnum,3,chirpnum,newframe-oldframe);
        TX1_DATA = squeeze(RX1_DATA(:,1,:,:));

        [vf,rf]=RealTimeProcess_STFT(TX1_DATA,adcnum,chirpnum,newframe-oldframe,range_min,range_max);

        %分类
        if size(vf,2)<40
            vf_con=[vf_con,vf];
            rf_con=[rf_con,rf];
            if size(vf_con,2)>40
                vf=vf_con;vf_con=[];
                rf=rf_con;rf_con=[];
            end
        end
        
        if size(vf,2)>40
            if size(vf_old,2)<40
                vf_old=[vf_old,vf];
                rf_old=[rf_old,rf];
            else
                vf_ansys=[vf_old,vf];
                rf_ansys=[rf_old,rf];
                idx_ansys=size(vf_ansys,2);
                labels=zeros(idx_ansys-79,1);
                for i_ansys=1:idx_ansys-79
                    vf_test=vf_ansys(:,[0:79]+i_ansys);
                    rf_test=rf_ansys(range_min:range_max,[0:79]+i_ansys);
                    feature=extractFeature_self(vf_test,rf_test);
                    labels(i_ansys) = predict(Mdl,feature(1:78));
                end
                label=mode(labels);
                vf_old=vf;
                rf_old=rf;
                ANSYS_ON=1;
            end
        end
        
        newframe_stft=oldframe_stft+(newframe-oldframe)*2;%补偿后
        if ANSYS_ON
            realtime=clock;
            toc_front=(oldframe_stft+1)*res_t;
            time_front(3)=starttime(6)+toc_front;
            time_front(3)=time_front(3)-60*(time_front(3)>60);
            time_front(2)=starttime(5)+1*(time_front(3)>60);
            time_front(1)=starttime(4)+1*(time_front(2)>60);

            toc_behind=(newframe_stft)*res_t;
            time_behind(3)=starttime(6)+toc_behind;
            time_behind(3)=time_behind(3)-60*(time_behind(3)>60);
            time_behind(2)=starttime(5)+1*(time_behind(3)>60);
            time_behind(1)=starttime(4)+1*(time_behind(2)>60);
            
            delay=realtime(6)-time_behind(3);
            delay=delay+60*(delay<0);

            disp(['REAL_T: ',num2str(realtime(4)),':',num2str(realtime(5)),':',num2str(realtime(6)),'    ',...
                'PROCESS_T:',num2str(time_front(1)),':',num2str(time_front(2)),':',num2str(time_front(3)),' ~ ',...
                num2str(time_behind(1)),':',num2str(time_behind(2)),':',num2str(time_behind(3)),'    ',...
                'RELATIVE_T:',num2str(toc_front),'s ~ ',num2str(toc_behind),'s    ',...
                'LABEL:',num2str(label),'  DELAY:',num2str(delay)]);

            ANSYS_ON=0;
        end


        % PLOT
        if PLOT_ON
            figure(1)
            subplot(1,2,1)%subplot(3,1,1)
            imagesc(([oldframe_stft+1:newframe_stft])*res_t,[1:size(rf,1)]*res_r,10*log10(abs(rf)));
            axis xy
            caxis([80,130]);
            xlabel("Time(s)");
            ylabel("Range(m)");
            title('Range');
            colormap('jet')
            axis([1*res_t newframe_stft*res_t range_min*res_r range_max*res_r]) % 1:20 range
            hold on

            subplot(1,2,2)%subplot(3,1,2)
            imagesc(([oldframe_stft+1:newframe_stft])*res_t,[-size(vf,1)/2:size(vf,1)/2-1]*res_v,10*log10(abs(vf)));
            axis xy
            caxis([80,130]);
            xlabel("Time(s)");
            ylabel("velocity(m/s)");
            title('Doppler');
            colormap('jet')
            axis([1*res_t newframe_stft*res_t -size(vf,1)/2*res_v (size(vf,1)/2-1)*res_v])
            hold on
            
%             if k>1
%             subplot(3,1,3)
%             color=[{'red'};{'red'};{'red'};{'red'};{'red'};{'red'}];
%             plot ([newframe_stft-idx_ansys+1,newframe_stft]*res_t,[label,label],color{label});
%             axis xy
%             xlabel("Time(s)");
%             ylabel("Act");
%             title('classification');
%             %axis([max(1,newframe_stft-159)*res_t newframe_stft*res_t -1 3])
%             axis([1*res_t newframe_stft*res_t 0 7])
%             hold on
%             else
%                 set(1,'outerposition',get(0,'screensize')); %窗口最大化
%             end
        end

        if USER_ON && k>1         
            figure(2)
            subplot(2,1,1)
            plot ([oldframe_stft+1,newframe_stft]*res_t,[label,label],'Color','red','LineWidth',2);
            hold on
            if k==2
                plot ([1,oldframe_stft]*res_t,[label,label],'Color','red','LineWidth',2);
                hold on
            end

            subplot(2,1,2)
            lightcontrol(label);
        end

        file(k,4)=toc;
        file(k,5)=file(k,4)-file(k,1);
        t=clock;
        timetable(k)=[num2str(t(4)),' : ',num2str(t(5)),' : ',num2str(t(6))];
        oldframe=newframe;
        oldframe_stft=newframe_stft;
        k=k+1;
    end
    
    if oldframe>340+conframe
        %outerbytes=framebytes-mod(filebytes,framebytes);%溢出的字节
        outerbytes=0;
        fclose('all');
        fileName=['D:\RadarProcessing\RawData\',saveName,'_Raw_',num2str(raw_idx),'.bin'];
        fid = fopen(fileName,'r');
        fseek(fid,outerbytes,'bof');
        raw_idx=raw_idx+1;
        conframe=oldframe;
        %close all
    end
        
end
fclose('all');
disp('采集完成');
% delete(['D:\RadarProcessing\RawData\',saveName,'_Raw_0.bin']);
% delete(['D:\RadarProcessing\RawData\',saveName,'_Raw_1.bin']);
% delete(['D:\RadarProcessing\RawData\',saveName,'_Raw_2.bin']);
delete(['D:\RadarProcessing\RawData\',saveName,'_LogFile.txt']);
delete(['D:\RadarProcessing\RawData\',saveName,'_Raw_LogFile.csv']);


