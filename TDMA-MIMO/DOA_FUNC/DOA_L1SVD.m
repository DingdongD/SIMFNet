function PoutSVD = DOA_L1SVD(X, A, P)
    % ������ΪL1-SVD�ĺ���ʵ���ļ�
    % X �������ź�
    % A �����걸��
    % P �� ��Դ��Ŀ
    
    [M, snap] = size(X); % M ��Ԫ snap ����
    thetaNum = size(A, 2); % ɨ���������
    DK1 = eye(P);
    DK2 = zeros(P, snap-P);
    DK = [DK1, DK2].'; % SNAP * P��ѡ�����
    [U, Sigm, V] = svd(X); % ����ֵ�ֽ�
    Xsv = X * V * DK; % ��ȡ�µĽ��վ���
    
    % ������Ȳ��Է���
    cvx_begin quiet
        variables p q
        variables r(thetaNum)
        variable SSV1(thetaNum, P) complex
        expressions Rn(M, P) complex

        minimize(p + 2.7 * q); % �Ż�Ŀ��
        subject to
            Rn = Xsv - A * SSV1; % ��в�
            Rvec = vec(Rn); % ����ת��Ϊ����
            norm(Rvec) <= p; % ��һ������ʽԼ��
            sum(r) <= q; % �ڶ�������ʽԼ��
            for i = 1 : thetaNum % ����������ʽԼ��
                norm(SSV1(i, :)) <= r(i);
            end
    cvx_end
    
    % ������Ȳ��Է���
    % confidence_interval = 0.9; % ����ֵ
    % noise = X - X0; % ��������
    % noise_var = var(noise(:)); % �����������
    % regulari_param = compute_regulariParam(confidence_interval, noise_var, M, P); % ���ݿ����ֲ���������ֵ
    % cvx_begin quiet
    %     variable SSV1(length(theta_grids), P) complex
    %     minimize(sum(norms(SSV1, 2, 2)))
    %     subject to
    %         norm(Xsv - A * SSV1, 'fro') <= regulari_param 
    % cvx_end
    PoutSVD = abs(SSV1(:, :)  / max(SSV1(:, :))); % ��⹦����
    
end