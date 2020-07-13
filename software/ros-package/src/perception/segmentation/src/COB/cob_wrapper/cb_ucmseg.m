function resp = cb_ucmseg(~, reqMsg, respMsg)
%CB_UCMSEG Summary of this function goes here
%   Detailed explanation goes here

img = readImage(reqMsg.Image);
img = im2uint8(img);
cob_params = set_params(img);

[~,ucms,~] = img2ucms(img, cob_params);

n_hiers = size(ucms,3);
for ii=1:n_hiers
    % Transform the UCM to a hierarchy
    curr_hier = ucm2hier(ucms(:,:,ii));

    respMsg.ImagesSeg(ii) = rosmessage('sensor_msgs/Image');
    respMsg.ImagesSeg(ii).Encoding = 'mono16';
    writeImage(respMsg.ImagesSeg(ii),uint16(curr_hier.leaves_part));
end

resp = respMsg;

end

