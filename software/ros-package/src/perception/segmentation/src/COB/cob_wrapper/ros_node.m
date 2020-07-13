%!source ../../../devel/setup.bash

if ~exist('install','file')
    addpath('../COB');
end

if ~exist('img2ucms','file')
    install;
end

addpath('../../matlab_gen/msggen');

rosinit;
masterHost = 'localhost';
node = robotics.ros.Node('cob_wrapper_node', masterHost);
srv  = robotics.ros.ServiceServer(node,'/segmentation/cob', 'cob_wrapper/cob_segmentation_srv', @cb_ucmseg);

if 0 % for test
    img = imread('~/tmp/table_1/000000.color.png');
    srvclient = rossvcclient('/segmentation/cob');
    req = rosmessage(srvclient);
    req.Image.Encoding = 'rgb8';
    writeImage(req.Image,img);
    res = call(srvclient,req,'Timeout',3);
    for i=1:numel(res.ImagesSeg)
        subplot(1,numel(res.ImagesSeg),i);
        img = readImage(res.ImagesSeg(i));
        imagesc(img);
    end
end

if 0 %shutdown
    clear 'node';
    rosshutdown;    
end
