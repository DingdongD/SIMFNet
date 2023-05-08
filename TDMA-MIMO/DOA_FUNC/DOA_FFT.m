function [PoutFFT] = DOA_FFT(arrData, cfgDOA)
    %% ���ļ�Ϊ����FFT��DOA/AOA����
    %% By Xuliang, 20230412
    
    doa_fft = fftshift(fft(arrData, cfgDOA.FFTNum)) * 2 / cfgDOA.FFTNum ;
    PoutFFT = (doa_fft);
    
end