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
    axes(handles.axes2); %ָ����Ҫ��յ�������
    cla reset;
    set(handles.axes2,'box','on')
    set(handles.axes2,'xtick',[]);
    set(handles.axes2,'ytick',[]);
    
    root_path = 'H:\RadarProcessing\DataFile\TestData\'; % ��·������
    data_name = get(handles.edit1,'string');
    data_path = strcat(root_path,data_name);
    if exist(data_path,'dir') == 0 % ������ļ��в������򴴽�һ����Ŀ¼
        mkdir(data_path);
    elseif ((flag == 1) & (exist(data_path,'dir') ~= 0)) % ������ļ��д�������һ������
        set(handles.text1,'string',"�ļ����Ѵ�����");
    end
    flag = 0; % ���״̬
    % �ж��ļ��������Ƿ���bin�ļ����� û����ɼ� ���򷵻ط�ֹ����
    
    %% �޸�Lua�ļ�
    str1 = strcat('adc_data_path="H:\\RadarProcessing\\DataFile\\TestData\\',data_name,'\\adc_data.bin"'); % ���bin�ļ�Ŀ¼
    str = [str1,"ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)","RSTD.Sleep(1000)","ar1.StartFrame()"];
    fid = fopen('C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Scripts\CaptureData1243.lua','w');
    for i = 1:length(str)
        fprintf(fid,'%s\n',str(i));
    end
    fclose(fid); % �ر��ļ�
    
    %% ����ΪRSTD����studio�ɼ�����
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
    
    pause(20); % 20s�������ݲɼ��Ļ���
    
    %% ����Ƿ�ɼ���ȷ����
    data_path = data_path + "\adc_data_Raw_0.bin";
    if exist(data_path,'file') == 0 % ��ʾû�вɼ�������
        set(handles.text1,'string',"δ��ȷ�ɼ�������");
    else
        D = dir(data_path);
        path_size = D.bytes / 1024;
        if path_size == 614400
            set(handles.text1,'string',"�ļ���ȷ�ɼ�");
        else
            set(handles.text1,'string',"�ļ���С����,�����²ɼ���");
        end
    end
              
% --- Executes on button press in button3.
function button3_Callback(hObject, eventdata, handles)
% hObject    handle to button3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    global flag;
    flag = 0;
    root_path = 'H:\RadarProcessing\DataFile\TestData\'; % ��·������
    data_name = get(handles.edit1,'string'); % �����ļ�������
    data_path = strcat(root_path,data_name);
    if exist(data_path,'dir') == 0 % ������ļ��в������򴴽�һ����Ŀ¼
        mkdir(data_path);
        set(handles.text1,'string',"�ļ������ɹ�");
        flag = 1;
    end
    filename = char(strcat("H:\RadarProcessing\DataFile\TestData\",data_name,"\video"));
%     set(handles.text1,'string',filename);
    
    axes(handles.axes1);
    closepreview;
    video_information = imaqhwinfo();
    % video_format = video_information.DeviceInfo.SupportedFormats; % �����ڲ鿴��ʽ
%     vid = videoinput('winvideo',2,'RGB32_1280x720');
%     vid = videoinput('winvideo',3,'RGB32_640x360');
    vid = videoinput('winvideo',1,'MJPG_1280x720');
    
%     winvideoinfo = imaqhwinfo('winvideo');
%     vid = videoinput('winvideo',1,'H264_1024x576');
    preview(vid)
    nframe = 640; % �м�֡����  
    nrate = 20; % ÿ��֡�ʱ�ʾÿ�뼸֡��nframe/nrate s
    MakeVideo(vid, filename, nframe, nrate, 2) % ��ʼ����ͼ������
    closepreview;
    
% --- Executes on button press in button4.
function button4_Callback(hObject, eventdata, handles)
% hObject    handle to button4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    set(handles.text1,'string'," ");
    pause(1);
    
    %% ����ļ������Ƿ����bin�ļ� 
    root_path = 'H:\RadarProcessing\DataFile\TestData\'; % ��·������
    data_name = get(handles.edit1,'string'); % �����ļ�������
    filename = strcat(root_path,data_name,'\adc_data_Raw_0.bin'); % �����ļ������Ƿ����bin�ļ�
    
    if exist(filename,'file') == 0 % ������ļ��в������򴴽�һ����Ŀ¼
       return; 
    end
    
    %% �״������������
    global RANGE_AXIS;
    global VELOCITY_AXIS;
    global Range_Time_Plot
    global Micro_Doppler_Plot
    global FRAME_AXIS;
    
    F0 = 77e9; % ��ʼƵ��
    c = physconst('lightspeed'); % ����
    LAMBDA = c / F0; % ����
    D = LAMBDA / 2; % ���߼��
    RX_NUM = 4; % ����������Ŀ
    TX_NUM = 3; % ����������Ŀ
    CHIRP_NUM = 128; % ÿ֡chirp����
    IDEL_TIME = 100e-6; % ����ʱ��
    RAMP_TIME = 60e-6; % �������ʱ��
    TC = (IDEL_TIME + RAMP_TIME) * TX_NUM; % ��֡ʱ��
    TF = TC * CHIRP_NUM; % ֡��ʱ��CHIRP
    SLOPE = 60.012e12; % ��Ƶб��
    ADC_SAMPLE = 512; % ADC������
    FS_SAMPLE = 10000e3; % ������
    FRAME_LENGTH = 200; % ֡��
    FRAME_AXIS = (1:FRAME_LENGTH)*80/1e3; % ֡ʱ��
    RANGE_RES = 3e8 / (2 * 1 / FS_SAMPLE * ADC_SAMPLE * SLOPE); % ����ֱ���
    RANGE_AXIS = [1: ADC_SAMPLE] * RANGE_RES; % ���뵥Ԫ
    VELOCITY_RES = 3e8 / (2 * F0 * TF); % �ٶȷֱ���
    VELOCITY_AXIS = [-CHIRP_NUM / 2 : CHIRP_NUM / 2 - 1] * VELOCITY_RES; % �ٶȵ�Ԫ
    
    [NewData] = Reshape_Data(filename,ADC_SAMPLE,TX_NUM,CHIRP_NUM,FRAME_LENGTH);
    [RANGE_PROFILE] = RangeFFT(NewData);
    [DOPPLER_PROFILE] = DopplerFFT(RANGE_PROFILE);
    Range_Time_Plot = (abs(sum(squeeze(DOPPLER_PROFILE(:,1,:,:)),2)));
    Micro_Doppler_Plot = (abs(sum(squeeze(DOPPLER_PROFILE(:,1,:,:)),1)));
    
     set(handles.text1,'string',"������ɣ�");
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
    axes(handles.axes2); %ָ����Ҫ��յ�������
    cla reset;
    set(handles.axes2,'box','on')
    set(handles.axes2,'xtick',[]);
    set(handles.axes2,'ytick',[]);
    axes(handles.axes1); %ָ����Ҫ��յ�������
    cla reset;
    set(handles.axes1,'box','on')
    set(handles.axes1,'xtick',[]);
    set(handles.axes1,'ytick',[]);


% --- Executes on button press in button7.
function button7_Callback(hObject, eventdata, handles)
    root_path = 'H:\RadarProcessing\DataFile\TestData\'; % ��·������
    data_name = get(handles.edit1,'string'); % �����ļ�������
    data_path = char(strcat(root_path,data_name));
    cd(data_path);
    delete *.txt;
    delete *.bin;
    delete *.csv;
    delete *.avi;
    set(handles.text1,'string',"�ļ������ϣ�");
    cd('H:\RadarProcessing\RadarCaptureSystem');

    
