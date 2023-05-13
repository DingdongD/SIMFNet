
global framevalue
framevalue=100;

figure(2)
subplot(2,1,1)
axis xy
xlabel("Time(s)");
ylabel("Act");
title('classification');
%axis([max(1,newframe_stft-159)*res_t newframe_stft*res_t -1 3])
t=framevalue*0.08;
plot ([0,t],[1,1],'k--',[0,t],[2,2],'k--',[0,t],[3,3],'k--',...
    [0,t],[4,4],'k--', [0,t],[5,5],'k--', [0,t],[6,6],'k--');
axis([0 t 0 7])
hold on

subplot(2,1,2)
pos1 = [2 4 1 1];
rectangle('Position',pos1,'Curvature',[1 1],'FaceColor','white')
text(2.1,3.7,'走路','FontSize',12);
text(2.35,4.5,'1','FontSize',12,'Color','white');

pos2 = [4 4 1 1];
rectangle('Position',pos2,'Curvature',[1 1],'FaceColor','white')
text(4.1,3.7,'坐立','FontSize',12);
text(4.35,4.5,'2','FontSize',12,'Color','white');

pos3 = [6 4 1 1];
rectangle('Position',pos3,'Curvature',[1 1],'FaceColor','white')
text(6.1,3.7,'跌起','FontSize',12);
text(6.35,4.5,'3','FontSize',12,'Color','white');

pos4 = [8 4 1 1];
rectangle('Position',pos4,'Curvature',[1 1],'FaceColor','white')
text(8.1,3.7,'挥手','FontSize',12);
text(8.35,4.5,'4','FontSize',12,'Color','white');

pos5 = [10 4 1 1];
rectangle('Position',pos5,'Curvature',[1 1],'FaceColor','white')
text(10.1,3.7,'拳击','FontSize',12);
text(10.35,4.5,'5','FontSize',12,'Color','white');

pos6 = [12 4 1 1];
rectangle('Position',pos6,'Curvature',[1 1],'FaceColor','white')
text(12.1,3.7,'无人','FontSize',12);
text(12.35,4.5,'6','FontSize',12,'Color','white');

text(4,6,'Real-Time Act Recognition Result','FontSize',12,'FontWeight','bold','Color','red');
axis([2 13 3.5 7])
axis off
hold on