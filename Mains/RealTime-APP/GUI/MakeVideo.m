function MakeVideo(vid,filename,nframe,N,Vformat)
    %% ���ļ����ڿ���¼��
    %% vid��video�������
    %% filename �� video����
    %% nframe �� video��֡��
    %% N �� ÿ��֡��
    %% Vformat �� 1-gray 2-rgb
    
    if Vformat == 1
        movieformat = 'grayscale';
        colorformat = 'gray';
    elseif Vformat == 2
        movieformat = 'rgb';
        colorformat = [];
    end
    
%     preview(vid); % ��ȡvideo����
    set(1,'visible','off'); % ��ֹͼ����ʾ
    set(vid,'ReturnedColorSpace',movieformat); % ������ɫ�ռ�ΪGRAY����RGB
    writerObj = VideoWriter(filename);
    writerObj.FrameRate = N;
    open(writerObj);
    
    for i = 1: nframe
        frame = getsnapshot(vid);
        imshow(frame);
        f.cdata = frame;
        f.colormap = colormap(colorformat) ;
        writeVideo(writerObj,f);
    end
    close(writerObj);
    closepreview;
end
    