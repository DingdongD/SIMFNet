function MakeVideo(vid,filename,nframe,N,Vformat)
    %% 本文件用于控制录像
    %% vid：video输入对象
    %% filename ： video名称
    %% nframe ： video的帧数
    %% N ： 每秒帧率
    %% Vformat ： 1-gray 2-rgb
    
    if Vformat == 1
        movieformat = 'grayscale';
        colorformat = 'gray';
    elseif Vformat == 2
        movieformat = 'rgb';
        colorformat = [];
    end
    
%     preview(vid); % 读取video输入
    set(1,'visible','off'); % 禁止图形显示
    set(vid,'ReturnedColorSpace',movieformat); % 设置颜色空间为GRAY还是RGB
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
    