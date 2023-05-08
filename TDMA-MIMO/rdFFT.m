function fftOut = rdFFT(adcData, IQFlag)
    %% ���ļ�����ʵ�־���ά��������άFFT
    %% By Xuliang,20230412
    
    ADCNum = size(adcData, 1);
    ChirpNum = size(adcData, 2);
    arrNum = size(adcData,3);
    fftOut = {};
    
    if IQFlag
        % ����άFFT
        rangeWin = hanning(ADCNum); % ����Ӵ�
        rangeWin3D = repmat(rangeWin, 1, ChirpNum, arrNum); % ������rangeData����һ��
        rangeData = adcData .* rangeWin3D ; % ����ά�Ӵ�
        rangeFFTOut = fft(rangeData, [], 1) * 2 * 2 / ADCNum; % �Ծ���ά��FFT��FFT����+������������
        
        % ������άFFT 
        dopplerWin = hanning(ChirpNum)'; % ������
        dopplerWin3D = repmat(dopplerWin, ADCNum, 1, arrNum); % ������dopplerData����һ��
        dopplerData = rangeFFTOut .* dopplerWin3D; % �����ռӴ�
        dopplerFFTOut = fftshift(fft(dopplerData, [], 2),2) * 2 * 2 / ChirpNum; % �Զ�����ά��FFT��FFT����+������������ 
    else
        % ����I·ʱ��Ҫע��Ŀ����벻�ܳ������Լ��Ŀ����� �������־���ģ��
        % ��·�ź���Ҫ�׵�һ���ź�

        % ����άFFT
        rangeWin = hanning(ADCNum); % ������
        rangeWin3D = repmat(rangeWin, 1, ChirpNum, arrNum); % ������rangeData����һ��
        rangeData = adcData .* rangeWin3D ; % ����ά�Ӵ�
        rangeFFTOut = fft(rangeData, [], 1) * 2 * 2 / ADCNum; % �Ծ���ά��FFT��FFT����+������������
        rangeFFTOut = rangeFFTOut(1:end/2, :, :);
        
        % ������άFFT 
        dopplerWin = hanning(ChirpNum)'; % ������
        dopplerWin3D = repmat(dopplerWin, ADCNum / 2, 1, arrNum); % ������dopplerData����һ��
        dopplerData = rangeFFTOut .* dopplerWin3D; % �����ռӴ�
        dopplerFFTOut = fftshift(fft(dopplerData, [], 2),2) * 2 * 2 / ChirpNum; % �Զ�����ά��FFT��FFT����+������������

    end
    fftOut.rangeFFT = rangeFFTOut;
    fftOut.dopplerFFT = dopplerFFTOut;
end