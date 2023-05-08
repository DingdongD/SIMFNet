function RAM = GenerateRAM(rangeFFTOut, cfgOut, cfgDOA)
    %% ���ļ�����ʵ�־���-��λͼ����
    %% ���ﶪ�����˶�����ά��Ϣ���洢����
    %% By Xuliang,20230421
    
    rangeNum = size(rangeFFTOut, 1); % ��ȡ���뵥Ԫ��Ŀ
    thetaGrids = cfgDOA.thetaGrids; % ����
    virtual_array = cfgOut.virtual_array; % ��������
    
    RAM = zeros(rangeNum, length(thetaGrids)); % ��ʼ��RAM
    for rangeIdx = 1 : rangeNum
       arrData = squeeze(rangeFFTOut(rangeIdx, :, :)); % �������� ChirpNum * arrNum
       sig = reshape(arrData, size(arrData,1), cfgOut.numRx, cfgOut.numTx); % ChirpNum * RXNum * TXNum 
       tmpSig = zeros(size(arrData,1), max(virtual_array.azi_arr)+1, max(virtual_array.ele_arr)+1); % ��ʼ���ź��ӿռ�
       for trx_id = 1 : size(cfgOut.sigIdx,2)
           tmpSig(:, cfgOut.sigSpaceIdx(1, trx_id), cfgOut.sigSpaceIdx(2,trx_id)) = sig(:, cfgOut.sigIdx(1,trx_id), cfgOut.sigIdx(2,trx_id)); % ���ź���źſռ�
       end
       aziData = squeeze(tmpSig(:, :, 1)); % ����ȡ��λͨ��
       [doaOut] = azimuthDOA(aziData.', cfgDOA);
       RAM(rangeIdx, :) = doaOut.spectrum; % �洢ÿ�����뵥Ԫ��Ӧ�Ŀռ���

    end
end
