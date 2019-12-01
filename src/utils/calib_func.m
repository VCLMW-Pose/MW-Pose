%%
% Copyright (c) 2019-2020, Shaoshu Yang All Right Reserved.
%
% This programme is developed as free software for the investigation of human 
% pose estimation using RF signals. Redistribution or modification of it is
% allowed under the terms of the GNU General Public Licence published by the
% Free Software Foundation, either version 3 or later version.
%
% Redistribution and use in source or executable programme, with or without 
% modification is permitted provided that the following conditions are met:
%
% 1. Redistribution in the form of source code with the copyright notice
%    above, the conditions and following disclaimer retained.
%
% 2. Redistribution in the form of executable programme must reproduce the
%    copyright notice, conditions and following disclaimer in the
%    documentation and\or other literal materials provided in the distribution.
%
% This is an unoptimized software designed to meet the requirements of the
% processing pipeline. No further technical support is guaranteed.

function y = calib_func(x)
    
% Acquire R-P-Y angles and translation from x
theta_x = x(1);     delta_x = x(4);
theta_y = x(2);     delta_y = x(5);
theta_z = x(3);     delta_z = x(6);

% Where homogeneous transformation matrix: 
% $T_{c}^{s}$ is: $$T_{c}^{s}=\left[\begin{matrix}
%     C_y C_z & C_z S_x S_y - C_x S_z & S_x S_z + C_x C_z S_y & \delta_x \\
%     C_y S_z & C_x C_z + S_x S_y S_z & C_x S_y S_z - C_z S_z & \delta_y \\
%     -C_y    & C_y S_x               & C_x C_y               & \delta_z \\
%     0       & 0                     & 0                     & 1
%                             \end{matrix}\right]$$
R = [cos(theta_y)*cos(theta_z)  ...
    cos(theta_z)*sin(theta_x)*sin(theta_y) - cos(theta_x)*sin(theta_z) ...
    sin(theta_x)*sin(theta_z) + cos(theta_x)*cos(theta_z)*sin(theta_y);
    cos(theta_y)*sin(theta_z) ...
    cos(theta_x)*cos(theta_z) + sin(theta_x)*sin(theta_y)*sin(theta_z) ...
    cos(theta_x)*sin(theta_y)*sin(theta_z) - cos(theta_z)*sin(theta_z);
    -sin(theta_y)   cos(theta_y)*sin(theta_x)   cos(theta_x)*cos(theta_y)];
T = [delta_x, delta_y, delta_z]';

% Read calibration points from file
calibPoint = load("calibPoint.mat");
calibPoint = calibPoint.calibPoint;
y = zeros(calibPoint.num*3, 1);

for i = 1:calibPoint.num
    
    % For every calibration point $P_{si}, P_{ci}$, the mapping is designated 
    % as: $$P_{si}=T_{c}^{s}P_{ci}$$
    y(3*(i - 1) + 1:3*i) = R*calibPoint.CameraCoordinates(i, :)' + T - ...
                                calibPoint.WalabotCoordinates(i, :)';
end
end