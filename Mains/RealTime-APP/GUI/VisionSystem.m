function varargout = VisionSystem(varargin)
% VISIONSYSTEM MATLAB code for VisionSystem.fig
%      VISIONSYSTEM, by itself, creates a new VISIONSYSTEM or raises the existing
%      singleton*.
%
%      H = VISIONSYSTEM returns the handle to a new VISIONSYSTEM or the handle to
%      the existing singleton*.
%
%      VISIONSYSTEM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in VISIONSYSTEM.M with the given input arguments.
%
%      VISIONSYSTEM('Property','Value',...) creates a new VISIONSYSTEM or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before VisionSystem_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to VisionSystem_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help VisionSystem

% Last Modified by GUIDE v2.5 23-Feb-2022 16:45:02

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @VisionSystem_OpeningFcn, ...
                   'gui_OutputFcn',  @VisionSystem_OutputFcn, ...
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


% --- Executes just before VisionSystem is made visible.
function VisionSystem_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to VisionSystem (see VARARGIN)

% Choose default command line output for VisionSystem
    handles.output = hObject;

% Update handles structure
    guidata(hObject, handles);

% UIWAIT makes VisionSystem wait for user response (see UIRESUME)
% uiwait(handles.figure1);
    set( handles.axes1,'box','on')
    set(handles.axes1,'xtick',[]);
    set(handles.axes1,'ytick',[]);

% --- Outputs from this function are returned to the command line.
function varargout = VisionSystem_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
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
    video_information = imaqhwinfo('winvideo');
    % video_format = video_information.DeviceInfo.SupportedFormats; % 可用于查看格式
%     vid = videoinput('winvideo',2,'RGB32_1280x720');
    vid = videoinput('winvideo',3,'RGB32_640x360');
%     vid = videoinput('winvideo',1,'MJPG_1280x720');
    
%     winvideoinfo = imaqhwinfo('winvideo');
%     vid = videoinput('winvideo',1,'H264_1024x576');
    preview(vid)
    nframe = 320; % 有几帧数据  
    nrate = 16; % 每秒帧率表示每秒几帧，nframe/nrate s
    MakeVideo(vid, filename, nframe, nrate, 2) % 开始生成图像数据
    closepreview;
    set(handles.text1,'string',"采集完毕");


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
