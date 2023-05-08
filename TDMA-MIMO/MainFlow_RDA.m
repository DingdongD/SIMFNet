%% ���ļ����ڴ���TDMA-MIMO�״��ź�
%% By Xuliang,20230411
clc;clear;close all;

dataPath = './dataset/adc_raw_dat.mat'; % �����ļ�·��
ShowIQ = 1; % �Ƿ���ʾIQ�ź�
ShowRange = 1; % �Ƿ���ʾRangePorfile
ShowRD = 1; % �Ƿ���ʾRD
ShowCFAR = 1; % �Ƿ���ʾCFAR���
ShowPeak = 1; % �Ƿ���ʾ�ۺϷ�ֵ���
IQFlag = 1; % �Ƿ�ѡ��IQ·�ź� 0-��· 1-˫·
saveFlag = 0; % �Ƿ񱣴��ļ�

%% Ŀ���ϵͳ��������
disp(strcat(['=====','ˢ��Ŀ���ϵͳ����','====='])); % �����״�ģʽ
tarOut = ConfigureTarget;     % ����Ŀ����Ϣ
cfgOut = ConfigureParameter;  % ���ɺ��ײ��״�ϵͳ����

%% �״�ɼ�����
disp(strcat(['=====','�״���빤��״̬','=====']));     
[RawData] = GenerateAdcData(tarOut, cfgOut, IQFlag, saveFlag, dataPath); % ��ʼģ�������

%% �����������
c = physconst('LightSpeed'); % ����
fc = cfgOut.fc; % ��Ƶ Hz
lambda = c / fc; % ����

ADCNum = cfgOut.ADCNum; % ADC������Ŀ
ChirpNum = cfgOut.ChirpNum; % ÿ֡����Chirp��Ŀ

numTx = cfgOut.numTx; % ����������Ŀ
numRx = cfgOut.numRx; % ����������Ŀ
% arr = cfgOut.array; % ��Ԫ����[���װ汾������]
virtual_array = cfgOut.virtual_array; % ��������struct
arr = virtual_array.virtual_arr;

arrNum = numTx * numRx; % ��Ԫ��Ŀ
arrDx = cfgOut.arrdx; % ��λ����Ԫ���
arrDy = cfgOut.arrdx; % ��������Ԫ���

validB = cfgOut.validB; % ��Ч����
range_res = c / (2 * validB); % ����ֱ���

TF = cfgOut.Tc * (numTx * ChirpNum);
doppler_res = lambda / (2 * TF); % �����շֱ���
Frame = cfgOut.Frame; % ֡��

% �����������ٶ�����
if IQFlag
    velocityIndex = [-ChirpNum / 2 : 1 : ChirpNum / 2 - 1] * doppler_res;
    rangeIndex = (0 : ADCNum - 1) * range_res;
else
    velocityIndex = [-ChirpNum / 2 : 1 : ChirpNum / 2 - 1] * doppler_res;
    rangeIndex = (-ADCNum / 2 : 1 : ADCNum / 2 - 1) * range_res;
    rangeIndex = rangeIndex(end/2+1:end);  % ȡ����һ��
end

%% IQƽ�����
if ShowIQ
    if IQFlag % IQ��·�ź�
        figure(1);
        set(gcf,'unit','centimeters','position',[10,12,10,10])
        plot(1:ADCNum, real(RawData(:,1,1))); hold on;
        plot(1:ADCNum, imag(RawData(:,1,1))); hold off;
        xlabel('ADC Points');ylabel('Amplitude');title('IQ-signal Analysis');
        legend('I-Chain','Q-Chain');grid minor;
    else
        figure(1);
        set(gcf,'unit','centimeters','position',[10,12,10,10])
        plot(1:ADCNum, (RawData(:,1,1)));
        xlabel('ADC Points');ylabel('Amplitude');title('IQ-signal Analysis');grid minor;
    end
end
RawData = reshape(RawData, ADCNum, ChirpNum, [], arrNum);
disp(strcat(['�״��źŵ�ά��Ϊ��',num2str(size(RawData))]));

for frame_id = 1 : Frame % Frame
    adcData = squeeze(RawData(:, :, frame_id, :));
    
    %% ����άFFT�Ͷ�����FFT
    disp(strcat(['=====','RD-MAP����','====='])); 
    tic
    fftOut = rdFFT(adcData, IQFlag);
    toc
    rangeFFTOut = fftOut.rangeFFT;
    dopplerFFTOut = fftOut.dopplerFFT;
    if ShowRange
        figure(2);
        set(gcf,'unit','centimeters','position',[10,0,10,10])
        plot(rangeIndex, db(rangeFFTOut(:,1,1))); 
        xlabel('Range(m)');ylabel('Amplitude(dB)'); 
        title(strcat(['��',num2str(frame_id),'֡-Ŀ�����ֲ�']));grid minor;  
        pause(0.1);
    end
    if ShowRD
        figure(3);
        set(gcf,'unit','centimeters','position',[20,12,10,10])
        imagesc(rangeIndex, velocityIndex, db(dopplerFFTOut(:,:,1).'));
        xlabel('Range(m)');ylabel('Velocity(m/s)'); colormap('jet');
        title(strcat(['��',num2str(frame_id),'֡Ŀ��-��������շֲ�']));
        grid minor; axis xy;
        pause(0.1);
    end

    %% ����ɻ���
    disp(strcat(['=====','����ɻ���','====='])); 
    RDM = dopplerFFTOut;
    tic
    [accumulateRD] = incoherent_accumulation(RDM);
    toc
    
    %% CFAR�����
    disp(strcat(['=====','���龯���','====='])); 
    Pfa = 1e-3; % �龯����
    TestCells = [8, 8]; % �ο���
    GuardCells = [2, 2]; % ������
    
    tic
    [cfarOut] = CFAR_2D(accumulateRD, Pfa, TestCells, GuardCells);
    toc
    cfarMap = cfarOut.cfarMap; % �������
    noiseOut = cfarOut.noiseOut; % �������
    snrOut = cfarOut.snrOut; % ��������
    
    if ShowCFAR
        figure(4);
        set(gcf,'unit','centimeters','position',[20,0,10,10])
        imagesc(rangeIndex, velocityIndex, cfarMap.');
        xlabel('Range(m)');ylabel('Velocity(m/s)'); colormap('jet');
        title(strcat(['��',num2str(frame_id),'֡Ŀ��-CFAR�����']));
        grid minor;axis xy;
        pause(0.1);
    end
    
    %% ��ֵ�ۺ�-��ȡ��Ŀ��
    disp(strcat(['=====','��ֵ�ۺ�','====='])); 
    [range_idx, doppler_idx] = find(cfarMap);
    cfar_out_idx = [range_idx doppler_idx]; % ��ȡCFAR����������������
    tic
    [rd_peak_list, rd_peak] = peakFocus(db(accumulateRD), cfar_out_idx);
    toc
    peakMap = zeros(size(cfarMap)); % ��ֵ�ۺϽ������
    for peak_idx = 1 :size(rd_peak_list, 2)
        peakMap(rd_peak_list(1,peak_idx), rd_peak_list(2,peak_idx)) = 1;
    end
    
    if ShowPeak
        figure(5);
        set(gcf,'unit','centimeters','position',[30,12,10,10])
        imagesc(rangeIndex, velocityIndex, peakMap.');
        xlabel('Range(m)');ylabel('Velocity(m/s)'); colormap('jet');
        title(strcat(['��',num2str(frame_id),'֡Ŀ��-��ֵ�ۺϽ��']));
        grid minor;axis xy;
        pause(0.1);
    end

    %% DOA/AOA����
    disp(strcat(['=====','DOA/AOA����','====='])); 
    cfgDOA.FFTNum = 180; % FFT����
    % ������Ȼ��װ�˲�ͬ��DOA�㷨 ������Ҫע����� ���õ��㷨���� �ڱ��״����� ����ʹ�õ���FFT-FFT��FFT-MUSIC��
    % ��ΪMUSIC-MUSIC��ʹ�ûᵼ���ڷ�λά�ȿռ��׹���ʱ�ƻ���λ�ź� ������ά�ȵ���λ����
    
    cfgDOA.AziMethod = 'FFT'; % ��λά��DOA���Ʒ���
    cfgDOA.EleMethod = 'MUSIC'; % ����ά��DOA���Ʒ���
    
    cfgDOA.thetaGrids = linspace(-90, 90, cfgDOA.FFTNum); % �ռ�����
    cfgDOA.AzisigNum = 1; % Լ��ÿ��RD-CELL�ϵ���Դ��Ŀ
    cfgDOA.ElesigNum = 1; % Լ��ÿ����λ�׷��ϵ���Դ��Ŀ
    
    aziNum = length(virtual_array.noredundant_aziarr); % ��λ������Ŀ
    
%     Aset = exp(-1j * 2 * pi * arrDx * [0:aziNum]' * sind(cfgDOA.thetaGrids)); % ϡ���ֵ����
    targetPerFrame = {}; 
    targetPerFrame.rangeSet = [];
    targetPerFrame.velocitySet = [];
    targetPerFrame.snrSet = [];
    targetPerFrame.azimuthSet = [];
    targetPerFrame.elevationSet = [];
    
    if ~isempty(rd_peak_list) % �ǿձ�ʾ��⵽Ŀ��
        rangeVal = (rd_peak_list(1, :) - 1) * range_res; % Ŀ�����
        speedVal = (rd_peak_list(2, :) - ChirpNum / 2 - 1) * doppler_res; % Ŀ���ٶ�
        
        doaInput = zeros(size(rd_peak_list, 2), arrNum);
        for tar_idx = 1 :size(rd_peak_list, 2)
            doaInput(tar_idx, :) = squeeze(dopplerFFTOut(rd_peak_list(1, tar_idx), rd_peak_list(2, tar_idx), :)); % tarNum * arrNum
        end
        doaInput = reshape(doaInput, [], numRx, numTx);
        
        % ��λ�ǹ���ǰ��Ҫ���Ƕ����ղ���
        [com_dopplerFFTOut] = compensate_doppler(doaInput, cfgOut, rd_peak_list(2, :), speedVal, rangeVal); 
        
        tic
        for peak_idx = 1 : size(rd_peak_list, 2) % ������⵽��ÿ��Ŀ��
            snrVal = mag2db(snrOut(rd_peak_list(1, peak_idx), rd_peak_list(2, peak_idx))); % ����ȵ�����������chirpNum*ADCNum�Ļ���
            tarData = squeeze(com_dopplerFFTOut(peak_idx, :,:));

    %         aziarr = unique(arr(1,arr(2,:)==0)); % ���װ汾��ȡ��λά����������
    %         aziArrData = arrData(aziarr+1); % ��ȡ��λά���ź�

            % ��λ�ǽ���
           sig = tarData;
           sig_space = zeros(max(virtual_array.azi_arr)+1,max(virtual_array.ele_arr)+1); % ��ʼ���ź��ӿռ�
           for trx_id = 1 : size(cfgOut.sigIdx,2)
               sig_space(cfgOut.sigSpaceIdx(1, trx_id), cfgOut.sigSpaceIdx(2,trx_id)) = sig(cfgOut.sigIdx(1,trx_id), cfgOut.sigIdx(2,trx_id)); % ���ź���źſռ�
           end
           % �����������ɵ��ź��ӿռ�ά��Ϊ ��λ����������Ŀ * ��������������Ŀ 

           eleArrData = zeros(cfgDOA.FFTNum, size(sig_space,2)); % ����ά������
            for ele_idx = 1 : size(sig_space, 2) % �����ȡ������Ϊ�����䲻ͬ�Ŀռ��׹��Ʒ���
                tmpAziData = sig_space(:, ele_idx);
                [azidoaOut] = azimuthDOA(tmpAziData, cfgDOA); % ��ȡ��һ�з�λά��������Ϣ���з�λ�ǹ��� 
                eleArrData(:, ele_idx) = azidoaOut.spectrum(:); % ���ռ���
            end

            for azi_peak_idx = 1 : length(azidoaOut.angleVal) % �Է�λά�ȼ����׷���м���
                tmpEleData = eleArrData(azidoaOut.angleIdx(azi_peak_idx), :).'; % ��ȡ�뷽λάĿ��������ź�
                [eledoaOut] = elevationDOA(tmpEleData, cfgDOA); % ���и����ǹ���

                % ����Ŀ��ľ��롢�����ա�����ȡ���λ�͸�����Ϣ
                aziVal = azidoaOut.angleVal; 
                eleVal = eledoaOut.angleVal;
                targetPerFrame.rangeSet = [targetPerFrame.rangeSet, rangeVal(peak_idx)];
                targetPerFrame.velocitySet = [targetPerFrame.velocitySet, speedVal(peak_idx)];
                targetPerFrame.snrSet = [targetPerFrame.snrSet, snrVal];
                targetPerFrame.azimuthSet = [targetPerFrame.azimuthSet,aziVal];
                targetPerFrame.elevationSet = [targetPerFrame.elevationSet,eleVal];
            end
        end
        toc
    
        %% �������� 
        disp(strcat(['=====','��������','====='])); 
        tic
        pcd_x = targetPerFrame.rangeSet .* cosd(targetPerFrame.elevationSet) .* sind(targetPerFrame.azimuthSet);
        pcd_y = targetPerFrame.rangeSet .* cosd(targetPerFrame.elevationSet) .* cosd(targetPerFrame.azimuthSet);
        pcd_z = targetPerFrame.rangeSet .* sind(targetPerFrame.elevationSet);
        PointSet = [pcd_x.', pcd_y.', pcd_z.', targetPerFrame.velocitySet.', targetPerFrame.snrSet.'];
        toc

        %% ���ƾ���
        eps = 1.1; % ����뾶
        minPointsInCluster = 3; % ������С������ֵ
        xFactor = 1;   % �����ƾ����󣬱�С��������С ��Բ
        yFactor = 1;   % �����ƽǶȱ�󣬱�С��������С ��Բ 
        figure(6);
        set(gcf,'unit','centimeters','position',[30,0,10,10])
        disp(strcat(['=====','���ƾ���','====='])); 
        tic
        [sumClu] = dbscanClustering(eps,PointSet,xFactor,yFactor,minPointsInCluster,frame_id); % DBSCAN����
        toc
    end

end
