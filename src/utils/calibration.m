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

%% Solve Least Square Esitimation of Calibration
solve_calib = 1;

if solve_calib
    
    % Solve LSE calibration
    P = fsolve(@calib_func, [0, 0, 0, 0, 0, 0]);
end

%% Calibration Pick Walabot Sensor Point Script
pick_point_sensor = 0;

if pick_point_sensor
    
    % Load energy reflected by voxel in the spaces, w h n are dimensions of
    % walabot sensor image.
    w = 46; h = 45; n = 59;
    img = zeros(w, h, n);
    
    % Load image from txt files.
    for i = 0:n - 1
        s = sprintf('E:/capture/dets%d.txt', i);
        tmp = load(s);
        tmp = fliplr(tmp);
        img(:, :, i + 1) = tmp;
    end
    
    % Find the maxima of reflected energy, and transfer the index to matrix
    % subscript.
    max_pos = find(img == max(img(:)));
    s = size(img);
    [x, y, z] = ind2sub(s, max_pos);
    max_pos = [x - 1, y - 1, z - 1];
    
    % Visualization threshold, only the voxel that its reflected energy is
    % greater than threshold will be displayed in visualization.
    mask = img > 0;
    sphere(5);
    % Draw sphere to visualize voxels with considerable reflection.
    [x, y, z] = sphere();
    
    % Get colour map.
    colors = jet(256);
    for i = 1:2:w
        for j = 1:2:h
            for k = 1:n
                
                % determine whether this voxel reflected greater energy
                % than threshold
                if mask(i, j, k) == 0
                    
                    continue;
                end
                
                len = size(x, 1);
                color = zeros(21, 21, 3);
                
                % The density of energy is indicated by colour of sphere,
                % from light blue to dark red.
                color(:,:,1) = ones(len)*colors(img(i, j, k), 1); % red
                color(:,:,2) = ones(len)*colors(img(i, j, k), 2); % green
                color(:,:,3) = ones(len)*colors(img(i, j, k), 3); % blue
                
                % Plot a sphere and hold on for further ploting.
                surf(2*x + i, 2*y + j, 2*z + k, color, 'FaceAlpha', 0.3);
                hold on;
                
            end
        end
    end
    
    % Disable shading of the plot, and set its boundaries.
    shading interp 
    axis([1, w, 1, h, 1, n]);
    
end


%% Calibration Point Picking Script
pick_point = 0;

if pick_point
    
    % Load camera parameters
    cameraPrams = load('cameraparameter.mat');

    % Read image
    img_path = 'F:/captureNov15/2/0207.jpg';
    img = imread(img_path);

    % Rotate image and eradicate distortion
    % [img, new] = undistortImage(img, cameraParams);
    img = permute(img, [2, 1, 3]);
    img = fliplr(img);

    % Display image
    imshow(img);
    impixelinfo;
    
end