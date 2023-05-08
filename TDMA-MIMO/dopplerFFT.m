function fftOut = dopplerFFT(rangeFFTOut)
    %% ���ļ�����ʵ�ֶ�����άFFT
    %% By Xuliang,20230412
    
    ADCNum = size(rangeFFTOut, 1);
    ChirpNum = size(rangeFFTOut, 2);
    arrNum = size(rangeFFTOut,3);
    fftOut = {};
    % ������άFFT 
    dopplerWin = hanning(ChirpNum)'; % ������
    dopplerWin3D = repmat(dopplerWin, ADCNum, 1, arrNum); % ������dopplerData����һ��
    dopplerData = rangeFFTOut .* dopplerWin3D; % �����ռӴ�
    dopplerFFTOut = fftshift(fft(dopplerData, [], 2),2) * 2 * 2 / ChirpNum; % �Զ�����ά��FFT��FFT����+������������ 
    fftOut.dopplerFFT = dopplerFFTOut;
end