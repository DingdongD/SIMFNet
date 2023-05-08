function [signal] = GenerateSigIQ(tarOut, cfgOut)
    %% ���ļ�����˫·ADC-IQ�ź�
    %% By Xuliang, 20230411
    tarNum = tarOut.nums; % Ŀ����Ŀ
    numTx = cfgOut.numTx; % ����������Ŀ
    numRx = cfgOut.numRx; % ����������Ŀ
    
    ADCNum = cfgOut.ADCNum; % ADC������Ŀ
    Frame = cfgOut.Frame; % ֡��
    ChirpNum = cfgOut.ChirpNum; % ÿ֡����Chirp��Ŀ
    TotalChirpNum = Frame * ChirpNum; % ��Chirp��Ŀ
    
    % �źŲ���
    fc = cfgOut.fc; % ��Ƶ Hz
    fs = cfgOut.fs; % ADC����Ƶ�� Hz 
    Ramptime = cfgOut.Ramptime; % ����ʱ��
    Idletime = cfgOut.Idletime; % ����ʱ��
    Slope = cfgOut.Slope; % Chirpб��
    validB = cfgOut.validB; % ʵ����Ч����
    totalB = cfgOut.totalB; % ��������
    Tc = cfgOut.Tc; % ��Chirp����
    ValidTc = cfgOut.ValidTc; % ��Chirp��ADC��Ч����ʱ��
    
    % Ӳ������
    Pt = cfgOut.Pt; % dbm ���书��
    Fn = cfgOut.Fn; % ����ϵ��
    Ls = cfgOut.Ls;  % ϵͳ���
    
    % ��������
    antennaPhase = cfgOut.antennaPhase; % ������λ
    virtual_array = cfgOut.virtual_array; % �������� ���������������struct virtual_arr�а����˷��䷽λ��λ�͸�����λ �������� ��������˳�� 
    arr = virtual_array.virtual_arr;
    
    % ���װ汾˳�������
%     arr = cfgOut.array; % ������Ԫ����
    arrNum = size(arr, 1); % ����������Ŀ
    arrdelay = zeros(1, arrNum) + Tc * reshape(repmat([0:numTx-1], numRx,1), 1, arrNum); % [0 0 0 0 1 1 1 1 2 2 2 2]
%     arrdelay = zeros(1, arrNum); % ��һ���൱�ڳ����� �������ǵ������߼�ʱ��
    
    
    delayTx = Tc * numTx; % ��������ά�Ȼ��۵�ʱ��
    arrdx = cfgOut.arrdx; % ��һ����Ԫ���
    arrdy = cfgOut.arrdy; 
    
    c = physconst('LightSpeed'); % ����
    lambda = c / fc; % ����
    Ts = 1 / fs; % ADC����ʱ��
    
    % �ź�ģ��
    signal = zeros(ADCNum, TotalChirpNum, arrNum); % �ź�
    
    noiseI = normrnd(0, 1, ADCNum, TotalChirpNum, arrNum); % ��̬����
    noiseQ = normrnd(0, 1, ADCNum, TotalChirpNum, arrNum); % ��̬����
%     noiseI = exprnd(1, ADCNum, TotalChirpNum, arrNum); % ָ���ֲ�����
%     noiseQ = exprnd(1, ADCNum, TotalChirpNum, arrNum); % ָ���ֲ�����
    for tarId = 1 : tarNum
        disp(strcat(['���ڻ�ȡĿ��',num2str(tarId),'������']));
        % Ŀ�����
        targetR = tarOut.range(tarId);    % ���� m
        targetV = tarOut.velocity(tarId); % �ٶ� m/s
        targetAzi = tarOut.Azi(tarId);    % Ŀ�귽λ�� ��
        targetEle = tarOut.Ele(tarId);    % Ŀ�긩���� ��
        targetRCS = tarOut.RCS(tarId);    % Ŀ��RCSֵ
        targetGt  = tarOut.Gt(tarId);     % �������� 
        targetGr  = tarOut.Gr(tarId);     % �������� 
        [tarSNR] = CalculateSNR(targetR, targetRCS, targetGt, targetGr, lambda, Pt, Fn, Ls, validB); % �����
        A = sqrt(2 * db2pow(tarSNR)); % �źŷ���
        
        targPhi0 = rand * 2 * pi; % ���������λ[0 2*pi]

        tempSigI = zeros(ADCNum, TotalChirpNum, arrNum); % I·�����ź�
        tempSigQ = zeros(ADCNum, TotalChirpNum, arrNum); % Q·�����ź�
        tempSig = zeros(ADCNum, TotalChirpNum, arrNum); % I·�����ź�
        
        for channelId = 1 : arrNum
            for chirpId = 1 : TotalChirpNum
                for adcId = 1 : ADCNum
                    % ��Ŀ��ʱ��
                    tarDelay = 2 * targetR / c + 2 * targetV * ((chirpId - 1) * delayTx + arrdelay(channelId) + adcId * Ts) / c; 
                    
                    % ������������
                    % ���װ汾��Ŀ����λ���
%                     tarPhi = targPhi0 + 2 * pi * (arr(1, channelId) * sind(targetAzi) * arrdx + ...
%                             arr(2, channelId) * sind(targetEle) * arrdy) + deg2rad(antennaPhase(channelId)); % ������Ԫ��ʼ�������λ��Ŀ�겨���

                    tarPhi = targPhi0 + 2 * pi * (arr(channelId) * sind(targetAzi) * arrdx + ...
                            arr(arrNum+channelId) * sind(targetEle) * arrdy) + deg2rad(antennaPhase(channelId)); % ���￼���˲�ͬ����˳��������λ

                    % ��Ƶ�źţ�exp[1j * 2 *pi * fc * tau + 2 * pi * S * t * tau - pi * tau * tau]  t�ķ�ΧΪ0-Tc tau�ķ�ΧΪt+nTc
                    tempSigI(adcId, chirpId, channelId) = A * cos(2 * pi * (fc * tarDelay + Slope * tarDelay * adcId * Ts - Slope * tarDelay * tarDelay / 2) + tarPhi);
                    tempSigQ(adcId, chirpId, channelId) = A * sin(2 * pi * (fc * tarDelay + Slope * tarDelay * adcId * Ts - Slope * tarDelay * tarDelay / 2) + tarPhi);
                    tempSig(adcId, chirpId, channelId) = tempSigI(adcId, chirpId, channelId) + 1j * tempSigQ(adcId, chirpId, channelId);
                end
            end
        end
        signal = signal + tempSig; % ��Ŀ���ź����
    end
    signal = signal - (noiseI + noiseQ); % ��������
%     signal = reshape(signal, size(signal,1), size(signal,2), numRx, numTx);
end