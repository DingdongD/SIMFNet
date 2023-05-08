function [tarSNR] = CalculateSNR(targetR, targetRCS, Gt, Gr, lambda, Pt, Fn, Ls, FsAdc)
%% ���ļ����ڷ�������Ŀ�����
%% By Xuliang, 20230411

    Pt_dB = Pt - 30; % ���书�� dB
    R = db(targetR) / 2; % ����
    lambda = db(lambda ^ 2) / 2; % ����
    const = db((4 * pi) ^ 3) / 2; % ������
    
    K = 1.380649e-23; % ������������ J/k 
    B = FsAdc; % �������� HZ
    T0 = 290; % ������ϵ��
    KBT = db(K * B * T0) / 2; 
    
    tarSNR = (Pt_dB + Gt + Gr + targetRCS + lambda) - (KBT + Fn + const + 4 * R + Ls);
    
end