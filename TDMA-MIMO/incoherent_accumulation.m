function [accumulateRD] = incoherent_accumulation(RDM)
    %% ���ļ�����ʵ�ַ���ɻ���
    %% By Xuliang,20230412
    % RDM��ADCNum * ChirpNum * ARRNum
    
    accumulateRD = squeeze(sum(abs(RDM), 3))/sqrt(size(RDM,3));
end