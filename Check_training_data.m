clear;
clc;
close all;

% Specify the path to your HDF5 file
h5_file_path = './data_for_training/SR_5x5_4x/UrbanLF-Big/000016.h5';  % Update this path accordingly

% Read the datasets in the HDF5 file
Lr_SAI_y = h5read(h5_file_path, '/Lr_SAI_y');  % Low-resolution patch
Hr_SAI_y = h5read(h5_file_path, '/Hr_SAI_y');  % High-resolution patch
Pb_SAI_y = h5read(h5_file_path, '/Pb_y');  % Probability map

% Check if the masks exist and read them
info = h5info(h5_file_path);
datasets = {info.Datasets.Name};

% Check if Pb_SAI_mask1_y exists
if ismember('Pb_SAI_mask1_y', datasets)
    Mask1 = h5read(h5_file_path, '/Pb_SAI_mask1_y');  % Load Mask1
    disp('Pb_SAI_mask1_y loaded.');
else
    disp('Pb_SAI_mask1_y does not exist in the HDF5 file.');
end

% Check if Pb_SAI_mask2_y exists
if ismember('Pb_SAI_mask2_y', datasets)
    Mask2 = h5read(h5_file_path, '/Pb_SAI_mask2_y');  % Load Mask2
    disp('Pb_SAI_mask2_y loaded.');
else
    disp('Pb_SAI_mask2_y does not exist in the HDF5 file.');
end

% Display the sizes of the loaded datasets
disp('Size of Lr_SAI_y (Low-Resolution Patch):');
disp(size(Lr_SAI_y));

disp('Size of Hr_SAI_y (High-Resolution Patch):');
disp(size(Hr_SAI_y));

disp('Size of Pb_SAI_y (Probability Map):');
disp(size(Pb_SAI_y));

% Display mask sizes
if exist('Pb_SAI_mask1_y', 'var')
    disp('Size of Pb_SAI_mask1_y:');
    disp(size(Mask1));
end

if exist('Pb_SAI_mask1_y', 'var')
    disp('Size of Pb_SAI_mask1_y:');
    disp(size(Mask2));
end

%% Check if there are any values in Pb_SAI_y other than 0, 0.5, or 1
unique_values = unique(Pb_SAI_y(:));  % Get all unique values in the probability map
unexpected_values = unique_values(~ismember(unique_values, [0, 0.5, 1]));  % Find unexpected values

% Display the unexpected values, if any
if isempty(unexpected_values)
    disp('All values in the probability map are 0, 0.5, or 1 as expected.');
else
    disp('Unexpected values found in the probability map:');
    disp(unexpected_values);
end

%% Step 1: Extract the central sub-view from Lr_SAI_y and Hr_SAI_y
% Assuming angular resolution is angRes = 5, the central sub-view is at the 3rd angular position
angRes = 5;
central_u = ceil(angRes / 2);  % Central angular position (u)
central_v = ceil(angRes / 2);  % Central angular position (v)

patchsize = 32;  % Example patch size

% Extract the central sub-view
Lr_Central_View = Lr_SAI_y((central_u - 1) * patchsize + 1 : central_u * patchsize, ...
                           (central_v - 1) * patchsize + 1 : central_v * patchsize);
                       
Hr_Central_View = Hr_SAI_y((central_u - 1) * patchsize + 1 : central_u * patchsize, ...
                           (central_v - 1) * patchsize + 1 : central_v * patchsize);


%% Step 3: Visualize the central sub-view and probability map
% Display the low-resolution and high-resolution central sub-views
figure;
subplot(2, 3, 1);
imshow(Lr_Central_View, []);
title('Low-Resolution Central View');

subplot(2, 3, 2);
imshow(Hr_Central_View, []);
title('High-Resolution Central View');

% Convert the probability patch into a colored image
% Assuming the probability map is for 3 classes, we map each class to RGB
% Create a color image based on the probabilities
[r, g, b] = deal(Pb_SAI_y(:,:,1), Pb_SAI_y(:,:,2), Pb_SAI_y(:,:,3));
Prob_RGB = cat(3, r, g, b);  % Combine the channels into an RGB image

subplot(2, 3, 3);
imshow(Prob_RGB);
title('Probability Map (Colored)');

%% Step 4: Visualize the Masks
% Mask1 and Mask2 visualization, if they exist
if exist('Mask1', 'var')    
    subplot(2, 3, 4);
    imshow(Mask1, []);
    title('Mask1 Central View');
end

if exist('Mask2', 'var')

    subplot(2, 3, 5);
    imshow(Mask2, []);
    title('Mask2 Central View');
end
