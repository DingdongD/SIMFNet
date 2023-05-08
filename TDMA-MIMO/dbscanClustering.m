function  [sumClu]=dbscanClustering(eps,Obj,xFactor,yFactor,minPointsInCluster,FrameIndx)
   
    % �����ж�׼��        (x-r0)^2/xFactor^2  + (y-y0)^2/yFactor^2 + (v-v0)^2 /vFactor^2 < eps^2   
    % maxClusters��        ��������
    % minPointsInCluster�� һ��cluster����С����
    % maxPoints��          һ��cluster������������
    % Obj��׼���ݽṹ       = [x,y,R,v��peakVal��SNR��aoaVar];
  
    %����
    epsilon2_= eps*eps; % ����뾶      
    numPoints =	size(Obj,1); % ����
    visited = zeros(numPoints,1); % ���ʾ������ڼ�¼�Ƿ����
    clusterId = 0; % Ŀ�����
    sumClu=[]; 
    
    colors = 'bgrcm';
    for i=1:numPoints
        
        if visited(i) == 0 %δ��ǵ� �����еĵ���Ϊ0
            visited(i) = 1; %ȡ��һ�����ĵ�q�����Ϊ1
            tempIdx = i; % ���ڱ�ǵ�ǰ��   
            x = Obj(i,1); % ���ڻ�ȡ��ǰ���x����
            y = Obj(i,2); % ���ڻ�ȡ��ǰ���y����

            numInEps=1; 
            for k=1:numPoints % �������е�
                if visited(k) == 0 % ״̬0  
                    summ = (Obj(k,1)-x)^2/xFactor^2+...
                         (Obj(k,2)-y)^2 /yFactor^2;
                    if summ < epsilon2_  % ����������׼����õ����ڴ�cluster
                        numInEps = numInEps+ 1; % �ھӽڵ���+1                  
                        tempIdx = [tempIdx k]; % �洢Ŀ���
                    end
                end
            end        
            
            if numInEps > minPointsInCluster % ����ھӽڵ���>Ҫ�����С�ھӽڵ���
                visited(i) = 1;   % ����i����Ϊ���ĵ�
                for in = 1:numInEps   % ��������������ĵ㣬�����Ϊ�ѷ���
                    visited(tempIdx(in)) = 1;   
                end
                
                next = 2;
                while next <= length(tempIdx) % ���η��ʱ�ǵĽڵ�
                    point_ref = tempIdx(next);
                    x = Obj(point_ref,1);
                    y = Obj(point_ref,2);
                    tempInd = [];
                    for ind=1:numPoints % ����ǽڵ���Ϊ���Ľڵ㣬�����е����
                        if visited(ind) == 0 % ״̬0  
                            summ = (Obj(ind,1)-x)^2/xFactor^2+...
                                 (Obj(ind,2)-y)^2/yFactor^2;                            
                            if summ < epsilon2_ % �鿴�Ƿ����е㶼����������
                               tempInd = [tempInd ind]; % �����򽫸õ����
                            end
                        end
                    end

                    if length(tempInd) > minPointsInCluster % ������ĵ�Ĵ��ڽڵ��ھ�Ҳ������С�������
                        visited(point_ref) = 1;  % ��Ǵ��ڽڵ�Ϊ���Ľڵ㣬��Ϊͬ��
                        numInEps = numInEps+ length(tempInd); % ������ڵ���Ŀ
                        tempIdx = [tempIdx tempInd]; 
                        for kk = 1:length(tempInd)
                            visited(tempInd(kk)) = 1;  % ���Ϊ�ѷ���
                        end
                    else
                        visited(point_ref) = -1; %��������С�����������Ϊ�Ǳ߽��
                    end
                    next = next+1;
                end
                
                tempClu = Obj(tempIdx,:); % obj = [ X,Y,h ,objSpeed,snr]; ��ȡ���ĵ��������Ϣ
                cluLength = size(tempIdx,2);
              
                for pp = 1:cluLength % ��ÿ����ѭ��
                    ind = tempIdx(pp); 
                    if  visited(ind) == 1 % ���ƺ��ĵ�
                         plot(tempClu(pp,1),tempClu(pp,2),'.','color', colors(mod(clusterId,length(colors))+1));
                         hold on;
                    elseif  visited(ind) == -1  % ���Ʊ߽��
                           plot(tempClu(pp,1),tempClu(pp,2),'*','color', colors(mod(clusterId,length(colors))+1));                    
                         hold on;
                    end
                end
                title(['����������*:�߽�� , o�����ĵ� , .:������)֡����',num2str(FrameIndx)]);
                xlabel('�㼣ˮƽλ�� �� m');
                ylabel('�㼣��ֱλ�� �� m');
                hold on;
            else
                visited(i) = -2; %������  
                cluLength = 1;
                plot(Obj(i,1),Obj(i,2),'r.');
                hold on;
            end
             
            if cluLength > 1 
                clusterId = clusterId+1; % New cluster ID
                output_IndexArray(tempIdx) = clusterId;
                sumClu(clusterId).numPoints = cluLength; % ���ڳ���
                sumClu(clusterId).x_mean = mean(tempClu(:,1));  % ���ڵ�ƽ��x����
                sumClu(clusterId).y_mean = mean(tempClu(:,2));  % ���ڵ�ƽ��y����
%                 sumClu(clusterId).z_mean = mean(tempClu(:,3));  % ���ڵ�ƽ��z����

               % ͨ��SNR��Ȩ
               sumClu(clusterId).x_SNR = (1./tempClu(:,end)')*tempClu(:,1)/sum(1./tempClu(:,end));   
               sumClu(clusterId).y_SNR = (1./tempClu(:,end)')*tempClu(:,2)/sum(1./tempClu(:,end)); 
%                sumClu(clusterId).z_SNR = (1./tempClu(:,end)')*tempClu(:,3)/sum(1./tempClu(:,end));

                % ��ȡ��ֵ
                [ ~,I] = max(1./tempClu(:,end)); % snr��ֵ
                tempx = tempClu(:,1);
                tempy = tempClu(:,2);
%                 tempz = tempClu(:,3);
          
                sumClu(clusterId).x_peak = tempx(I);    
                sumClu(clusterId).y_peak = tempy(I);
%                 sumClu(clusterId).z_peak = tempz(I);

                % yȡ���λ�� xȡ��ֵ
                sumClu(clusterId).x_edge = mean(tempClu(:,1));   
                sumClu(clusterId).y_edge = min(tempClu(: ,2));
%                 sumClu(clusterId).z_edge = mean(tempClu(:,3));
     
                sumClu(clusterId).x = sumClu(clusterId).x_mean; % ��ȡƽ��x����  
                sumClu(clusterId).y = sumClu(clusterId).y_mean; % ��ȡƽ��y����  
%                 sumClu(clusterId).z = sumClu(clusterId).z_mean; % ��ȡƽ��z����  
        
                sumClu(clusterId).xsize = max(abs(tempClu(:,1) - sumClu(clusterId).x));
                sumClu(clusterId).ysize = max(abs(tempClu(:,2) - sumClu(clusterId).y));
%                 sumClu(clusterId).zsize = max(abs(tempClu(:,3) - sumClu(clusterId).z));
                sumClu(clusterId).v = mean(tempClu(:,end-1)); % �ٶ�
                sumClu(clusterId).head =  sumClu(clusterId).y - sumClu(clusterId).ysize/2;

                if sumClu(clusterId).xsize < 1e-3
                    sumClu(clusterId).xsize = 1;
                end
                if sumClu(clusterId).ysize < 1e-3
                    sumClu(clusterId).ysize = 1;
                end

                if cluLength>1
                   sumClu(clusterId).centerRangeVar = var(sqrt(tempClu(:,1).^2+tempClu(:,2).^2)); % ���뷽��
                   sumClu(clusterId).centerAngleVar = mean(deg2rad(atand(tempClu(:,2)/(tempClu(:,1)+eps))).^2); 
                   sumClu(clusterId).centerDopplerVar = var(tempClu(:,end-1)); % �����շ���

                else
                   sumClu(clusterId).centerRangeVar = 1;
                   sumClu(clusterId).centerAngleVar = 1;
                   sumClu(clusterId).centerDopplerVar = 1;
                end
            else
               continue;
            end
 
         end
    end   

    grid on
%     xlim([-100,100])
%     ylim([0,100])
    hold off
end