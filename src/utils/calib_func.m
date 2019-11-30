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
T = [cos(theta_y)*cos(theta_z)  ...
    cos(theta_z)*sin(theta_x)*sin(theta_y) - cos(theta_x)*sin(theta_z) ...
    sin(theta_x)*sin(theta_z) + cos(theta_x)*cos(theta_z)*sin(theta_y) ...
    delta_x;
    cos(theta_y)*sin(theta_z) ...
    cos(theta_x)*cos(theta_z) + sin(theta_x)*sin(theta_y)*sin(theta_z) ...
    cos(theta_x)*sin(theta_y)*sin(theta_z) - cos(theta_z)*sin(theta_z) ...
    delta_y;
    -cos(theta_y)   cos(theta_y)*sin(theta_x)   cos(theta_x)*cos(theta_y)   delta_z;
    0               0                           0                           1];
    
% For every calibration point $P_{si}, P_{ci}$, the mapping is designated 
% as: $$P_{si}=T_{c}^{s}P_{ci}$$
    
    % Set value of beta and theta
    beta = x(1);
    theta = x(2);
    
    %                0.7071S(beta)S(theta) - 0.5C(beta) + 0.5C(beta)C(theta)
    % tan(gamma) =  -------------------------------------------------------
    %                                  0.5C(theta) + 0.5
    y(1) = 1.414*sin(beta)*sin(theta) - cos(beta) + cos(beta)*cos(theta) - ...
        tan(gamma)*(cos(theta) + 1);
    
    %                0.5S(beta)C(theta) - 0.7071C(beta)S(theta) - 0.5S(beta)
    % tan(delta) -  -------------------------------------------------------
    %                                  0.5C(theta) + 0.5                  
    y(2) = sin(beta)*cos(theta) - 1.414*cos(beta)*sin(theta) - sin(beta) - ...
        tan(delta)*(cos(theta) + 1);
end