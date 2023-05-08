%% �״﷽�̣�������ĳһĿ��RCSֵ�²�ͬ�����µ�SNR
clear all; close all; clc;


%���������Լ���λת��
P_t = 12;       %dBm   10*log10(P/1mw)    [����˵���Ǵ�PA�����Ĺ��ʣ���ʵ���ᾭ���������������ٸ������ߣ�ʵ�ʷ����Ŀ��ܻ�С�ڸ�ֵ]
G_t = 12;       %dBi   10*log10(P/1)
G_r = 12;       %dBi    
RCS = 10;       %dBsm  10*log10(RCS)      

C      = 3e8;
fre    = 77e9;     %Hz
lambda = C/fre;

nums_rangefft   = 256;
nums_dopplerfft = 128;
G_s             = nums_dopplerfft*nums_rangefft;   
% G_s = 1;
 
K   = 1.380649e-23;   %J/k      ������������
T_0 = 290;            %k       
B   = 1.72e9;           %Hz      20e6 6874e3 
F   = 12;             %dB
L   = 3;              %dB     


%����λת������Ҫͳһ��dBת���ɷ���
P_t = (10^(P_t/10))/1e3;    %PtӦ�ñ��w
G_t = (10^(G_t/10));
G_r = (10^(G_r/10));
RCS = (10^(RCS/10));
F   = (10^(F/10));     
L   = (10^(L/10));



%�����Լ�������ͬ�����µ�SNR
R   = (0:0.1:180);
SNR = P_t * G_t * G_r * RCS * lambda^2 * G_s ./ (K * T_0 * B * F * ((4*pi)^3) * (R.^4) * L);
SNR = 10*log10(SNR);
figure(1)
plot(R,SNR);
title(['SNR������ϵ���� ', 'RCS = ',num2str(10*log10(RCS)),'dBsm']);xlabel('range(m)');ylabel('SNR(dB)');
grid on;





