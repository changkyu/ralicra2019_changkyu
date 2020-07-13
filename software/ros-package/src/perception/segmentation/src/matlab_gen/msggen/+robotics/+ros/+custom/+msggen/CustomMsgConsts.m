classdef CustomMsgConsts
    %CustomMsgConsts This class stores all message types
    %   The message types are constant properties, which in turn resolve
    %   to the strings of the actual types.
    
    %   Copyright 2014-2017 The MathWorks, Inc.
    
    properties (Constant)
        cob_wrapper_cob_segmentation_srv = 'cob_wrapper/cob_segmentation_srv'
        cob_wrapper_cob_segmentation_srvRequest = 'cob_wrapper/cob_segmentation_srvRequest'
        cob_wrapper_cob_segmentation_srvResponse = 'cob_wrapper/cob_segmentation_srvResponse'
        realsense_camera_StreamSensor = 'realsense_camera/StreamSensor'
        realsense_camera_StreamSensorRequest = 'realsense_camera/StreamSensorRequest'
        realsense_camera_StreamSensorResponse = 'realsense_camera/StreamSensorResponse'
        segmentation_Segmentation_srv = 'segmentation/Segmentation_srv'
        segmentation_Segmentation_srvRequest = 'segmentation/Segmentation_srvRequest'
        segmentation_Segmentation_srvResponse = 'segmentation/Segmentation_srvResponse'
    end
    
    methods (Static, Hidden)
        function messageList = getMessageList
            %getMessageList Generate a cell array with all message types.
            %   The list will be sorted alphabetically.
            
            persistent msgList
            if isempty(msgList)
                msgList = cell(6, 1);
                msgList{1} = 'cob_wrapper/cob_segmentation_srvRequest';
                msgList{2} = 'cob_wrapper/cob_segmentation_srvResponse';
                msgList{3} = 'realsense_camera/StreamSensorRequest';
                msgList{4} = 'realsense_camera/StreamSensorResponse';
                msgList{5} = 'segmentation/Segmentation_srvRequest';
                msgList{6} = 'segmentation/Segmentation_srvResponse';
            end
            
            messageList = msgList;
        end
        
        function serviceList = getServiceList
            %getServiceList Generate a cell array with all service types.
            %   The list will be sorted alphabetically.
            
            persistent svcList
            if isempty(svcList)
                svcList = cell(3, 1);
                svcList{1} = 'cob_wrapper/cob_segmentation_srv';
                svcList{2} = 'realsense_camera/StreamSensor';
                svcList{3} = 'segmentation/Segmentation_srv';
            end
            
            % The message list was already sorted, so don't need to sort
            % again.
            serviceList = svcList;
        end
        
        function actionList = getActionList
            %getActionList Generate a cell array with all action types.
            %   The list will be sorted alphabetically.
            
            persistent actList
            if isempty(actList)
                actList = cell(0, 1);
            end
            
            % The message list was already sorted, so don't need to sort
            % again.
            actionList = actList;
        end
    end
end
