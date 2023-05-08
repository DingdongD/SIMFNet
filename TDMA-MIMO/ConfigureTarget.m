function [tarOut] = ConfigureTarget()
    %% ���ļ����ڷ�������Ŀ�����
    %% By Xuliang, 20230411
    
    tarOut.nums = 4; % Ŀ����Ŀ
    
    % Ŀ�����: ���� �ٶ� ��λ ���� RCS Ŀ������
    tarOut.range = [5 10 15 18]; % ���� m
    tarOut.velocity = [0.4 -0.3 0 0.6]; % �ٶ� m/s
    tarOut.Azi = [15 -2 30 -25]; % Ŀ�귽λ�Ƕ�
    tarOut.Ele = [0 15 -5 8];     % Ŀ�긩���Ƕ�
    tarOut.RCS = [20 20 10 20];  % Ŀ��RCSֵ
    tarOut.Gt  = [12 12 12 12];  % ��������
    tarOut.Gr  = [12 12 12 13];  % ��������
    tarOut.trueX = tarOut.range .* cosd(tarOut.Ele) .* sind(tarOut.Azi);
    tarOut.trueY = tarOut.range .* cosd(tarOut.Ele) .* cosd(tarOut.Azi);
    tarOut.trueZ = tarOut.range .* sind(tarOut.Ele);
    
end