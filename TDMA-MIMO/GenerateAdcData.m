function [adcData] = GenerateAdcData(tarOut, cfgOut, IQFlag, saveFlag, dataPath)
    %% IQFlag: �����ź�ģʽ���
    %% saveFlag: ���������ݽ��б���
    %% ���ļ�����˫·ADC-IQ�ź�
    %% By Xuliang, 20230411
    
    % ADC��������
    if IQFlag
        tic
        adcData = GenerateSigIQ(tarOut, cfgOut);
        toc
    else
        tic
        [adcData] = GenerateSigI(tarOut, cfgOut);
        toc
    end
    disp(strcat(['=====','ADC�ź��������','====='])); 
    
    if saveFlag
        save(dataPath, 'adcData');
    end
end