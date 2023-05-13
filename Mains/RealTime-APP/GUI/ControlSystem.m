function varargout = ControlSystem(varargin)
    gui_Singleton = 1;
    gui_State = struct('gui_Name',       mfilename, ...
                       'gui_Singleton',  gui_Singleton, ...
                       'gui_OpeningFcn', @ControlSystem_OpeningFcn, ...
                       'gui_OutputFcn',  @ControlSystem_OutputFcn, ...
                       'gui_LayoutFcn',  [] , ...
                       'gui_Callback',   []);
    if nargin && ischar(varargin{1})
        gui_State.gui_Callback = str2func(varargin{1});
    end

    if nargout
        [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
    else
        gui_mainfcn(gui_State, varargin{:});
    end

function ControlSystem_OpeningFcn(hObject, eventdata, handles, varargin)
    handles.output = hObject;
    guidata(hObject, handles);
 
    set( handles.axes1,'box','on')
    set(handles.axes1,'xtick',[]);
    set(handles.axes1,'ytick',[]);
    set( handles.axes2,'box','on')
    set(handles.axes2,'xtick',[]);
    set(handles.axes2,'ytick',[]);


function varargout = ControlSystem_OutputFcn(hObject, eventdata, handles) 
    varargout{1} = handles.output;

function button1_Callback(hObject, eventdata, handles)
    addpath(genpath('.\'))
    RSTD_DLL_Path = 'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';
    ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);
    if (ErrStatus ~= 30000)
        disp('Error inside Init_RSTD_Connection');
        return;
    end
    strFilename ='C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\DataCaptureDemo_1243_3.lua';
    Lua_String = sprintf('dofile("%s")',strFilename);
    ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);


function button2_Callback(hObject, eventdata, handles)
    global flag;
    set(handles.text1,'string'," ");
    pause(1);
    axes(handles.axes2); %指定需要清空的坐标轴
    cla reset;
    set(handles.axes2,'box','on')
    set(handles.axes2,'xtick',[]);
    set(handles.axes2,'ytick',[]);
    
    root_path = 'H:\RadarProcessing\DataFile\TestData\'; % 根路径名称
    data_name = get(handles.edit1,'string');
    data_path = strcat(root_path,data_name);
    if exist(data_path,'dir') == 0 % 如果该文件夹不存在则创建一个该目录
        mkdir(data_path);
    elseif ((flag == 1) & (exist(data_path,'dir') ~= 0)) % 如果该文件夹存在则换另一个名称
        set(handles.text1,'string',"文件夹已创建！");
    end
    flag = 0; % 清除状态
    % 判断文件夹下面是否有bin文件存在 没有则采集 否则返回防止覆盖
    
    %% 修改Lua文件
    str1 = strcat('adc_data_path="H:\\RadarProcessing\\DataFile\\TestData\\',data_name,'\\adc_data.bin"'); % 设计bin文件目录
    str = [str1,"ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)","RSTD.Sleep(1000)","ar1.StartFrame()"];
    fid = fopen('C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Scripts\CaptureData1243.lua','w');
    for i = 1:length(str)
        fprintf(fid,'%s\n',str(i));
    end
    fclose(fid); % 关闭文件
    
    %% 下面为RSTD控制studio采集数据
    addpath(genpath('.\'))
    % Initialize mmWaveStudio .NET connection
    RSTD_DLL_Path = 'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';
    ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);
    if (ErrStatus ~= 30000)
        disp('Error inside Init_RSTD_Connection');
        return;
    end
    strFilename = 'C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\CaptureData1243.lua';
    Lua_String = sprintf('dofile("%s")',strFilename);
    ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
    
    pause(20); % 20s用于数据采集的缓冲
    
    %% 检测是否采集正确数据
    data_path = data_path + "\adc_data_Raw_0.bin";
    if exist(data_path,'file') == 0 % 表示没有采集到数据
        set(handles.text1,'string',"未正确采集到数据");
    else
        D = dir(data_path);
        path_size = D.bytes / 1024;
        if path_size == 614400
            set(handles.text1,'string',"文件正确采集");
        else
            set(handles.text1,'string',"文件大小错误,请重新采集！");
        end
    end
              
% --- Executes on button press in button3.
function button3_Callback(hObject, eventdata, handles)
% hObject    handle to button3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    global flag;
    flag = 0;
    root_path = 'H:\RadarProcessing\DataFile\TestData\'; % 根路径名称
    data_name = get(handles.edit1,'string'); % 读入文件夹名称
    data_path = strcat(root_path,data_name);
    if exist(data_path,'dir') == 0 % 如果该文件夹不存在则创建一个该目录
        mkdir(data_path);
        set(handles.text1,'string',"文件创建成功");
        flag = 1;
    end
    filename = char(strcat("H:\RadarProcessing\DataFile\TestData\",data_name,"\video"));
%     set(handles.text1,'string',filename);
    
    axes(handles.axes1);
    closepreview;
    video_information = imaqhwinfo();
    % video_format = video_information.DeviceInfo.SupportedFormats; % 可用于查看格式
%     vid = videoinput('winvideo',2,'RGB32_1280x720');
%     vid = videoinput('winvideo',3,'RGB32_640x360');
    vid = videoinput('winvideo',1,'MJPG_1280x720');
    
%     winvideoinfo = imaqhwinfo('winvideo');
%     vid = videoinput('winvideo',1,'H264_1024x576');
    preview(vid)
    nframe = 640; % 有几帧数据  
    nrate = 20; % 每秒帧率表示每秒几帧，nframe/nrate s
    MakeVideo(vid, filename, nframe, nrate, 2) % 开始生成图像数据
    closepreview;
    
% --- Executes on button press in button4.
function button4_Callback(hObject, eventdata, handles)
% hObject    handle to button4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    set(handles.text1,'string'," ");
    pause(1);
    
    %% 检查文件夹下是否存在bin文件 
    root_path = 'H:\RadarProcessing\DataFile\TestData\'; % 根路径名称
    data_name = get(handles.edit1,'string'); % 读入文件夹名称
    filename = strcat(root_path,data_name,'\adc_data_Raw_0.bin'); % 检测该文件夹下是否存在bin文件
    
    if exist(filename,'file') == 0 % 如果该文件夹不存在则创建一个该目录
       return; 
    end
    
    %% 雷达基本参数配置
    global RANGE_AXIS;
    global VELOCITY_AXIS;
    global Range_Time_Plot
    global Micro_Doppler_Plot
    global FRAME_AXIS;
    
    F0 = 77e9; % 起始频率
    c = physconst('lightspeed'); % 光速
    LAMBDA = c / F0; % 波长
    D = LAMBDA / 2; % 天线间距
    RX_NUM = 4; % 接收天线数目
    TX_NUM = 3; % 发射天线数目
    CHIRP_NUM = 128; % 每帧chirp个数
    IDEL_TIME = 100e-6; % 空闲时间
    RAMP_TIME = 60e-6; % 脉冲持续时间
    TC = (IDEL_TIME + RAMP_TIME) * TX_NUM; % 单帧时间
    TF = TC * CHIRP_NUM; % 帧间时间CHIRP
    SLOPE = 60.012e12; % 调频斜率
    ADC_SAMPLE = 512; % ADC采样点
    FS_SAMPLE = 10000e3; % 采样率
    FRAME_LENGTH = 200; % 帧数
    FRAME_AXIS = (1:FRAME_LENGTH)*80/1e3; % 帧时间
    RANGE_RES = 3e8 / (2 * 1 / FS_SAMPLE * ADC_SAMPLE * SLOPE); % 距离分辨率
    RANGE_AXIS = [1: ADC_SAMPLE] * RANGE_RES; % 距离单元
    VELOCITY_RES = 3e8 / (2 * F0 * TF); % 速度分辨率
    VELOCITY_AXIS = [-CHIRP_NUM / 2 : CHIRP_NUM / 2 - 1] * VELOCITY_RES; % 速度单元
    
    [NewData] = Reshape_Data(filename,ADC_SAMPLE,TX_NUM,CHIRP_NUM,FRAME_LENGTH);
    [RANGE_PROFILE] = RangeFFT(NewData);
    [DOPPLER_PROFILE] = DopplerFFT(RANGE_PROFILE);
    Range_Time_Plot = (abs(sum(squeeze(DOPPLER_PROFILE(:,1,:,:)),2)));
    Micro_Doppler_Plot = (abs(sum(squeeze(DOPPLER_PROFILE(:,1,:,:)),1)));
    
     set(handles.text1,'string',"处理完成！");
     clear NewData DOPPLER_PROFILE RANGE_PROFILE
     
function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    

% --- Executes on button press in box1.
function box1_Callback(hObject, eventdata, handles)
% Hint: get(hObject,'Value') returns toggle state of box1
    
    global RANGE_AXIS;
    global Range_Time_Plot
    global FRAME_AXIS
    if get(handles.box1,'Value') == 1
        axes(handles.axes2);
        imagesc(FRAME_AXIS,RANGE_AXIS,db(squeeze((Range_Time_Plot)))); % RANGE_AXIS,
        xlabel('Frame Period(s)');ylabel('Range(m)');
    end
    clear Range_Time_Plot RANGE_AXIS 

% --- Executes on button press in box2.
function box2_Callback(hObject, eventdata, handles)
% Hint: get(hObject,'Value') returns toggle state of box2
    global Micro_Doppler_Plot
    global VELOCITY_AXIS
    global FRAME_AXIS
    if get(handles.box2,'Value') == 1
        axes(handles.axes2);
        imagesc(FRAME_AXIS,VELOCITY_AXIS,db(squeeze(Micro_Doppler_Plot)));
        xlabel('Frame Period(s)');ylabel('Velocity(m/s)');
%         axis([min(FRAME_AXIS) max(FRAME_AXIS) min(VELOCITY_AXIS) max(VELOCITY_AXIS)]);

    end
    clear Micro_Doppler_Plot RANGE_AXIS VELOCITY_AXIS


% --- Executes on button press in button5.
function button5_Callback(hObject, eventdata, handles)
% hObject    handle to button5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    axes(handles.axes2); %指定需要清空的坐标轴
    cla reset;
    set(handles.axes2,'box','on')
    set(handles.axes2,'xtick',[]);
    set(handles.axes2,'ytick',[]);
    axes(handles.axes1); %指定需要清空的坐标轴
    cla reset;
    set(handles.axes1,'box','on')
    set(handles.axes1,'xtick',[]);
    set(handles.axes1,'ytick',[]);


% --- Executes on button press in button7.
function button7_Callback(hObject, eventdata, handles)
    root_path = 'H:\RadarProcessing\DataFile\TestData\'; % 根路径名称
    data_name = get(handles.edit1,'string'); % 读入文件夹名称
    data_path = char(strcat(root_path,data_name));
    cd(data_path);
    delete *.txt;
    delete *.bin;
    delete *.csv;
    delete *.avi;
    set(handles.text1,'string',"文件清空完毕！");
    cd('H:\RadarProcessing\RadarCaptureSystem');

    
