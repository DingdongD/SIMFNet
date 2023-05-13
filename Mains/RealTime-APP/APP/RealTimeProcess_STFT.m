function [vf,rf]=RealTimeProcess_STFT(TX1_DATA,adcnum,chirpnum,numframe,range_min,range_max)
temp=reshape(TX1_DATA,256,[]);

vf=zeros(256,2*numframe);
rf=zeros(256,2*numframe);

%% 只取1m的处理
vnonoise=[range_min:range_max];% 需要修改距离范围
for i=1:2*numframe-1
    adc=temp(:,[1:255]+127*(i-1));
    adc=bsxfun(@minus,adc,mean(adc));
    adc=bsxfun(@times,adc,hanning(adcnum));
    rfft=fft(adc,adcnum,1);
    %rfft=bsxfun(@times,rfft,hamming(256));
    rfft = rfft - (repmat(mean(rfft'),size(rfft,2),1))';%去0速度
    
    dfft=fft(rfft,256,2);%这里用256代替255
    dfft=fftshift(dfft,2);
    
    dsum=abs(dfft).^2;
    rsum=zeros(size(dsum,1),1);
    rsum(vnonoise)=sum(dsum(vnonoise,:),2);
    
    %plot(log10(abs(dsum)));
    dfft=abs(dfft(vnonoise,:)).^2;%3-4
    %dfft=abs(dfft(20:162,:)).^2;
    vf(:,i)=sum(dfft).';
    rf(:,i)=rsum;
end
    vf(:,2*numframe)=vf(:,2*numframe-1);
    rf(:,2*numframe)=rf(:,2*numframe-1);


%补0速度
vf(129,:)=mean(vf([127,128,130,131],:));

end

