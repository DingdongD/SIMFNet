% function [PoutANM,u_vec,T] = DOA_ANM(Y, P)
%     % ������Ϊvanilla-ԭ�ӷ����ĺ���ʵ���ļ�
%     % Y �������ź�
%     % A �����걸��
%     % P �� ��Դ��Ŀ
%     f0 = 77e9; % Ƶ��
%     c = 3e8; % ����
%     lambda = c / f0; % ����
%     d = lambda / 2; % ��Ԫ���
%     [M, snap] = size(Y); % ��Ԫ ����
%     
%     if snap == 1 % ������ģ��
%         sigma = 1; % ��������
%         regular_param = sqrt(M * log(M * sigma));
%         cvx_begin sdp quiet
%         cvx_solver sdpt3
%             variable T(M, M) hermitian toeplitz
%             variable x
%             variable z(M,1) complex
%             minimize (regular_param * 0.5 *(x + T(1,1)) + 0.5 * norm(Y-z))
%             [x Y'; Y T] >= 0;
%         cvx_end
%         [Phi, Val] = rootmusic(T, P, 'corr');
%         Phis = Phi / 2 / pi ;
%         estimated_theta = asind(-Phis * lambda / d);
%     else % �����ģ��
%         regular_param = sqrt(M * (snap + log(M) + sqrt(2 * snap * log(M))));
%         cvx_begin sdp quiet
%         cvx_solver sdpt3
%             variable T(M,M) hermitian toeplitz
%             variable X(snap, snap) hermitian
%             variable Z(M, snap) complex
%             minimize (regular_param * (trace(X) + trace(T)) + 1 / 2 * sum_square_abs(vec(Y - Z)));
%             [X Y';Y T] >= 0;
%         cvx_end
%         
%         [Phi, Val] = rootmusic(T, P, 'corr'); % ��u�лָ�����λֵ
%         Phis = Phi / 2 / pi ;
%         estimated_theta = asind(-Phis * lambda / d);
%     end
%     PoutANM = estimated_theta.';
%     u_vec = zeros(M, 1); % �����¶Խ�
%     for iid = 1 : M % ʹ��iȥ����ÿ���Խ���
%         for pid = iid : M
%             u_vec(iid) = u_vec(iid) + T(pid , pid - iid + 1); % pid ������ iid=3 pid=3:8 pid-iid+1=1:6 
%         end
%         u_vec(iid) = u_vec(iid) / (M - iid + 1);
%     end
% end
%    

function PoutANM = DOA_ANM(Y, P)
    % ������Ϊvanilla-ԭ�ӷ����ĺ���ʵ���ļ�
    % Y �������ź�
    % A �����걸��
    % P �� ��Դ��Ŀ
    f0 = 77e9; % Ƶ��
    c = 3e8; % ����
    lambda = c / f0; % ����
    d = lambda / 2; % ��Ԫ���
    [M, snap] = size(Y); % ��Ԫ ����
    
    if snap == 1 % ������ģ��
        sigma = 1; % ��������
        regular_param = sqrt(M * log(M * sigma));
        cvx_begin sdp quiet
        cvx_solver sdpt3
            variable T(M, M) hermitian toeplitz
            variable x
            variable z(M,1) complex
            minimize (regular_param * 0.5 *(x + T(1,1)) + 0.5 * norm(Y-z))
            [x Y'; Y T] >= 0;
        cvx_end
%         [Phi, Val] = rootmusic(T, P, 'corr');
%         Phis = Phi / 2 / pi ;
%         estimated_theta = asind(-Phis * lambda / d);
    else % �����ģ��
        regular_param = sqrt(M * (snap + log(M) + sqrt(2 * snap * log(M))));
        cvx_begin sdp quiet
        cvx_solver sdpt3
            variable T(M,M) hermitian toeplitz
            variable X(snap, snap) hermitian
            variable Z(M, snap) complex
            minimize (regular_param * (trace(X) + trace(T)) + 1 / 2 * sum_square_abs(vec(Y - Z)));
            [X Y';Y T] >= 0;
        cvx_end
%         [Phi, Val] = rootmusic(T, P, 'corr'); % ��u�лָ�����λֵ
%         Phis = Phi / 2 / pi ;
%         estimated_theta = asind(-Phis * lambda / d);
    end
%     PoutANM = estimated_theta;
    PoutANM = T;
end
    