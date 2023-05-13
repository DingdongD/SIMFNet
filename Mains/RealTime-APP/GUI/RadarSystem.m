function varargout = RadarSystem(varargin)
% RADARSYSTEM MATLAB code for RadarSystem.fig
%      RADARSYSTEM, by itself, creates a new RADARSYSTEM or raises the existing
%      singleton*.
%
%      H = RADARSYSTEM returns the handle to a new RADARSYSTEM or the handle to
%      the existing singleton*.
%
%      RADARSYSTEM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RADARSYSTEM.M with the given input arguments.
%
%      RADARSYSTEM('Property','Value',...) creates a new RADARSYSTEM or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RadarSystem_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RadarSystem_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RadarSystem

% Last Modified by GUIDE v2.5 22-Feb-2022 14:28:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RadarSystem_OpeningFcn, ...
                   'gui_OutputFcn',  @RadarSystem_OutputFcn, ...
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
% End initialization code - DO NOT EDIT


% --- Executes just before RadarSystem is made visible.
function RadarSystem_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RadarSystem (see VARARGIN)

% Choose default command line output for RadarSystem
    handles.output = hObject;

% Update handles structure
    guidata(hObject, handles);

% UIWAIT makes RadarSystem wait for user response (see UIRESUME)
% uiwait(handles.figure1);
    set( handles.axes2,'box','on')
    set(handles.axes2,'xtick',[]);
    set(handles.axes2,'ytick',[]);
    set( handles.axes3,'box','on')
    set(handles.axes3,'xtick',[]);
    set(handles.axes3,'ytick',[]);


% --- Outputs from this function are returned to the command line.
function varargout = RadarSystem_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in button1.
function button1_Callback(hObject, eventdata, handles)
% hObject    handle to button1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    addpath(genpath('.\'))
    RSTD_DLL_Path = 'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';
    ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);
    if (ErrStatus ~= 30000)
        disp('Error inside Init_RSTD_Connection');
        return;
    end
    strFilename ='C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\DataCaptureDemo_1243_7.lua';
    Lua_String = sprintf('dofile("%s")',strFilename);
    ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);

% --- Executes on button press in button2.
function button2_Callback(hObject, eventdata, handles)
% hObject    handle to button2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    set(handles.text1,'string'," ");
    pause(0.1);
    axes(handles.axes2); %ָ����Ҫ��յ�������
    cla reset;
    set(handles.axes2,'box','on')
    set(handles.axes2,'xtick',[]);
    set(handles.axes2,'ytick',[]);
    axes(handles.axes3); %ָ����Ҫ��յ�������
    cla reset;
    set(handles.axes3,'box','on')
    set(handles.axes3,'xtick',[]);
    set(handles.axes3,'ytick',[]);
    
    root_path = 'H:\RadarProcessing\DataFile\sea_data\'; % ��·������
    data_name = get(handles.edit1,'string');
    data_path = strcat(root_path,data_name);
    if exist(data_path,'dir') == 0 % ������ļ��в������򴴽�һ����Ŀ¼
        mkdir(data_path);
    elseif ((exist(data_path,'dir') ~= 0)) % ������ļ��д�������һ������
        set(handles.text1,'string',"�ļ����Ѵ�����");
    end
    
    F0 = 77e9; % ��ʼƵ��
    c = physconst('lightspeed'); % ����
    LAMBDA = c / F0; % ����
    D = LAMBDA / 2; % ���߼��
    TX_NUM = 3; % ����������Ŀ
    
    % ��Ӿ�ص�ʵ����� ��Զ11m �ٶ�3m/s �ű�5
%     SLOPE = 46.397e12; % ��Ƶб��
%     ADC_SAMPLE = 256; % ADC������
%     FS_SAMPLE = 6874e3; % ������
%     IDEL_TIME = 30e-6; % ����ʱ��
%     RAMP_TIME = 80e-6; % �������ʱ��
    
%     % ��������Ϊ15m������ٶ�Ϊ4m/s �ű�6
%     SLOPE = 17.173e12; % ��Ƶб��
%     ADC_SAMPLE = 256; % ADC������
%     FS_SAMPLE = 3437e3; % ������
%     IDEL_TIME = 3.50e-6; % ����ʱ��
%     RAMP_TIME = 77.90e-6; % �������ʱ��    
%     CHIRP_NUM = 128; % ÿ֡chirp���� 

    % ��������Ϊ50m������ٶ�Ϊ4m/s �ű�7
    SLOPE = 37.465e12; % ��Ƶб��
    ADC_SAMPLE = 128; % ADC������
    FS_SAMPLE = 2000e3; % ������
    IDEL_TIME = 5e-6; % ����ʱ��
    RAMP_TIME = 73.16e-6; % �������ʱ��    
    
    CHIRP_NUM = 192; % ÿ֡chirp���� lua5��128
    TC = (IDEL_TIME + RAMP_TIME) * TX_NUM; % ��֡ʱ��
    TF = TC * CHIRP_NUM; % ֡��ʱ��CHIRP
    RANGE_RES = 3e8 / (2 * 1 / FS_SAMPLE * ADC_SAMPLE * SLOPE); % ����ֱ���
    RANGE_AXIS = [-ADC_SAMPLE / 2 : ADC_SAMPLE / 2-1] * RANGE_RES; % ���뵥Ԫ
%     RANGE_AXIS = [1: ADC_SAMPLE/2] * RANGE_RES; % ���뵥Ԫ
    VELOCITY_RES = 3e8 / (2 * F0 * TF); % �ٶȷֱ���
    VELOCITY_AXIS = [-CHIRP_NUM / 2 : CHIRP_NUM / 2 - 1] * VELOCITY_RES; % �ٶȵ�Ԫ

    % �ж��ļ��������Ƿ���bin�ļ����� û����ɼ� ���򷵻ط�ֹ����
    filename = strcat(root_path,data_name,'\adc_data_Raw_0.bin'); % �����ļ������Ƿ����bin�ļ�
    SendCaptureCMD(data_name);
    pause(0.01);
    while(true)
        D = dir(filename);
        path_size = D.bytes / 1024; % �ļ���С
        if path_size ~= 0 
            [data] = readDCA1000(filename);  
            max_frame = floor(size(data,2)/(ADC_SAMPLE*TX_NUM*CHIRP_NUM)); 
            RX1_DATA = reshape(data(1,1:ADC_SAMPLE*TX_NUM*CHIRP_NUM*max_frame),ADC_SAMPLE,TX_NUM,CHIRP_NUM,max_frame);
            FRAME_SET = (1:max_frame)*100/1e3;
            TX1_DATA = squeeze(RX1_DATA(:,1,:,:));
            range_plane = zeros(ADC_SAMPLE/2,max_frame);
            micro_doppler = zeros(CHIRP_NUM,max_frame);
            for frame_idx = 1:max_frame
                adc_data = squeeze(TX1_DATA(:,:,frame_idx));
                adc_data = adc_data - mean(adc_data,1); % �˳���̬�Ӳ�
            %     adc_data = adc_data .* hanning(adc_sample); % �Ӵ�
                range_profile = fft(adc_data,ADC_SAMPLE,1); % ����fft 
                range_profile = range_profile - repmat(mean(range_profile'),size(range_profile,2),1)'; % �˳��ٶ�Ϊ0Ŀ��
                doppler_profile = fftshift(fft(range_profile,CHIRP_NUM,2),2); % ������fft

            %     cfar_matrix = cfar_ca_2d(doppler_profile,Tr,Td,Gr,Gd,alpha);
            %     doppler_profile(cfar_matrix == 0) = 0;
                dsum = abs(doppler_profile).^2;
                rsum = sum(dsum(1:end/2,:),2);
                vsum = sum(dsum(1:end/2,:),1);
                range_plane(:,frame_idx) = rsum;
                micro_doppler(:,frame_idx) = vsum;
            end
            axis xy;
            imagesc(FRAME_SET,RANGE_AXIS(end/2+1:end),db(abs(range_plane))/2,'Parent',handles.axes2);
            xlabel(handles.axes2,'Frame Period(s)');ylabel(handles.axes2,'Range(m)');
            colormap(jet);caxis([80 110]);
            axis xy;
            imagesc(FRAME_SET,VELOCITY_AXIS,(db(abs(micro_doppler)))/2,'Parent',handles.axes3);
            xlabel(handles.axes3,'Frame Period(s)');ylabel(handles.axes3,'Velocity(m/s)');
            colormap(jet);caxis([80 110]);
            pause(0.01);   
        end
        if path_size == 115200
            break; % ����ļ���С����ɼ�Ҫ�����˳�ѭ��
        end
    end
    clear RX1_DATA data RANGE_PROFILE DOPPLER_PROFILE
    set(handles.text1,'string',"�������");
    
% --- Executes on button press in button4.
function button4_Callback(hObject, eventdata, handles)
% hObject    handle to button4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    root_path = 'H:\RadarProcessing\DataFile\ZH\'; % ��·������
    data_name = get(handles.edit1,'string'); % �����ļ�������
    data_path = char(strcat(root_path,data_name));
    cd(data_path);
    delete *.txt;
    delete *.bin;
    delete *.csv;
    delete *.avi;
    set(handles.text1,'string',"�ļ������ϣ�");
    cd('H:\RadarProcessing\RadarCaptureSystem');

% --- Executes on button press in button5.
function button5_Callback(hObject, eventdata, handles)
% hObject    handle to button5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    axes(handles.axes2); %ָ����Ҫ��յ�������
    cla reset;
    set(handles.axes2,'box','on');
    set(handles.axes2,'xtick',[]);
    set(handles.axes2,'ytick',[]);

    axes(handles.axes3); %ָ����Ҫ��յ�������
    cla reset;
    set(handles.axes3,'box','on')
    set(handles.axes3,'xtick',[]);
    set(handles.axes3,'ytick',[]);
    
    set(handles.text1,'string',"������ϣ�");

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

function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double

% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
