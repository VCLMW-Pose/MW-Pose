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
solve_calib = 0;

if solve_calib
    
    % Solve LSE calibration
    options = optimoptions('fsolve','StepTolerance', 1e-9, 'MaxIterations', 1e5, 'MaxFunctionEvaluations', ...
            1e5,'Display', ...
            'iter','PlotFcn',@optimplotfval);
    
    P = fsolve(@calib_func, [0, 0, 0, 0, 0, 0], options);
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

%% Browse Radio Signal Routine
browse_signal = 1;

if browse_signal
    
    % Directory to RF signals, must be absolute directory
    fileFolder = fullfile("F:/captureNov15/3/signals");
    
    % Read all file names within the directory
    dirOutput = dir(fullfile(fileFolder,'0*'));
    fileNames = {dirOutput.name};
    
    % Get colour map.
    colors = jet(256);
    
    % Traverse files
    for n = 1:size(fileNames, 2)
        
        % Read file name
        fileName = fullfile(fileFolder, sprintf("%s", char(fileNames(n))));
        
        % Read RF signal in binary formation
        signalFile = fopen(fileName, 'rb');
        data = fread(signalFile, 'int32');
        fclose(signalFile);
        
        % Find dimension of signal
        w = data(1); h = data(2); d = data(3);
        
        % Reshape signal and rotate
        signal = data(4:end);
        signal = reshape(signal, d, h, w);
        signal = permute(signal, [3, 2, 1]);
        signal = fliplr(signal);
        
        % Visualization threshold, only the voxel that its reflected energy is
        % greater than threshold will be displayed in visualization.
        mask = signal > 0;
        sphere(5);
        
        % Draw sphere to visualize voxels with considerable reflection.
        [x, y, z] = sphere();
        len = size(x, 1);
        a = figure('Visible', 'off');
        for i = 1:2:w
            for j = 1:2:h
                for k = 1:d

                    % determine whether this voxel reflected greater energy
                    % than threshold
                    if mask(i, j, k) == 0

                        continue;
                    end
                    
                    % The density of energy is indicated by colour of sphere,
                    % from light blue to dark red.
                    color = repmat(reshape(colors(signal(i, j, k), :), 1, 1, 3), len, len, 1);

                    % Plot a sphere and hold on for further ploting.
                    surf(2 * x + i, 2 * y + j, 2 * z + k, color, 'FaceAlpha', 0.3);
                    hold on;

                end
            end
        end
        
        % Disable shading of the plot, and set its boundaries.
        shading interp 
        title(fileName);
        view(0, -90)
        axis([1, w, 1, h, 1, d]);
        print(a, '-dbmp', sprintf('image/%d', n))
        
        % Wait for keyboard press
%         while waitforbuttonpress ~= 1
%         end
        clf;
        
    end
    
%     save_dir = "result.gif";
%     for j = 1:n
%         A = imread(sprintf('image/%d.bmp', j));
%         [I, map] = rgb2ind(A,256);
% 
%         if(j == 1)
%             imwrite(I, map, save_dir, 'DelayTime', 0.05, 'LoopCount', Inf)
%         else
%             imwrite(I, map, save_dir, 'WriteMode', 'append', 'DelayTime', 0.05)    
%         end
%     end

end

%% Calibration Point Picking Script
pick_point = 0;

if pick_point
    
    % Load camera parameters
    % cameraPrams = load('cameraparameter.mat');

    % Read image
    img_path = 'F:/captureNov15/1/0013.jpg';
    img = imread(img_path);

    % Rotate image and eradicate distortion
    img = permute(img, [2, 1, 3]);
    img = fliplr(img);
    [img, new] = undistortImage(img, cameraParams);

    % Display image
    imshow(img);
    impixelinfo;
    
end
