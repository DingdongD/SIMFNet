function ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path)
% 本文件用于和mmWaveStudio建立连接

    if (strcmp(which('RtttNetClientAPI.RtttNetClient.IsConnected'),''))
        % 在打开MATLAB后首先运行本代码
        disp('Adding RSTD Assembly');
        RSTD_Assembly = NET.addAssembly(RSTD_DLL_Path);
        if ~strcmp(RSTD_Assembly.Classes{1},'RtttNetClientAPI.RtttClient')
            disp('RSTD Assembly not loaded correctly. Check DLL path');
            ErrStatus = -10;
            return
        end
        Init_RSTD_Connection = 1;
    elseif ~RtttNetClientAPI.RtttNetClient.IsConnected() 
        Init_RSTD_Connection = 1;
    else
        Init_RSTD_Connection = 0;
    end

    if Init_RSTD_Connection
        disp('Initializing RSTD client');
        ErrStatus = RtttNetClientAPI.RtttNetClient.Init();
        if (ErrStatus ~= 0)
            disp('Unable to initialize NetClient DLL');
            return;
        end
        disp('Connecting to RSTD client');
        ErrStatus = RtttNetClientAPI.RtttNetClient.Connect('127.0.0.1',2777);
        if (ErrStatus ~= 0)
            disp('Unable to connect to mmWaveStudio');
            disp('Reopen port in mmWaveStudio. Type RSTD.NetClose() followed by RSTD.NetStart()');
            return;
        end
        pause(1);
    end
    disp('Sending test message to RSTD');
    Lua_String = 'WriteToLog("Running script from MATLAB\n", "green")';
    ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
    if (ErrStatus ~= 30000)
        disp('mmWaveStudio Connection Failed');
    end
    disp('Test message success');
end
