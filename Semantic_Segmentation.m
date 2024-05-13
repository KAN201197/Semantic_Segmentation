% Load the DeepLabv3 model
loadedData = load('C:\Users\kenta\Documents\NUS All Taken Course Document\Autonomous Mobile Robotics (AMR) Course\Assignment from Atharva\me5413_homework1\me5413\2_segmentation\deeplabv3plusResnet18CamVid\deeplabv3plusResnet18CamVid.mat');
model = loadedData.net;

classes = string(model.Layers(end).Classes);

% Step 3: Perform image segmentation
sampleImagesPath = 'C:\Users\kenta\Documents\NUS All Taken Course Document\Autonomous Mobile Robotics (AMR) Course\Assignment from Atharva\me5413_homework1\me5413\2_segmentation\images'; % Replace with the path to the folder containing the sample images
sampleImages = dir(fullfile(sampleImagesPath, '*.jpg')); % Assuming the sample images are in JPG format

segmentedResults = cell(numel(sampleImages), 1);

% Define custom colors for each class
classColors = [
    0.5020    0.5020    0.5020   % Sky
    0.5020         0         0   % Building
    0.7529    0.7529    0.7529   % Pole
    0.5020    0.2510    0.5020   % Road
    0.2353    0.1569    0.8706   % Pavement
    0.5020    0.5020         0   % Tree
    0.7529    0.5020    0.5020   % SignSymbol
    0.2510    0.2510    0.5020   % Fence
    0.2510         0    0.5020   % Car
    0.2510    0.2510         0   % Pedestrian
    0         0    0.5020];       % Bicyclist

for i = 1:numel(sampleImages)
    % Read the input image
    imagePath = fullfile(sampleImagesPath, sampleImages(i).name);
    inputImage = imread(imagePath);
    inputSize = model.Layers(1).InputSize;
    I = imresize(inputImage, inputSize(1:2));
    
    % Perform image segmentation
    segmentedImg = semanticseg(I, model);
    
    % Modify the network's output classes and visualize the segmentation results
    [~, colorLabels] = camvidColorMap();
    segOverlay = labeloverlay(I, segmentedImg, 'ColorMap', classColors, 'Transparency', 0.4);
    
    % Display the segmented image
    figure;
    imshow(segOverlay);
    
    % Create custom colorbar legend
    colorbar('off'); % Turn off default colorbar
    
    % Plot custom colorbar legend
    colormap(classColors);
    caxis([1 size(classColors, 1)]);
    h = colorbar;
    set(h, 'YTick', 1:size(classColors, 1), 'YTickLabel', colorLabels);
    ylabel(h, 'Class Labels');
    
    % Save the segmented image
    [~, imageName, ~] = fileparts(imagePath);
    savePath = fullfile('C:\Users\kenta\Documents\NUS All Taken Course Document\Autonomous Mobile Robotics (AMR) Course\Assignment from Atharva\me5413_homework1\me5413\2_segmentation\Result', [imageName, '_segmented.jpg']);
    imwrite(segOverlay, savePath);
    
    % Store the segmentation result for further analysis
    segmentedResults{i} = segmentedImg;
end

function [cmap, colorLabels] = camvidColorMap()

% Define the colormap used by CamVid dataset.
cmap = [
    128 128 128   % Sky
    128 0 0       % Building
    192 192 192   % Pole
    128 64 128    % Road
    60 40 222     % Pavement
    128 128 0     % Tree
    192 128 128   % SignSymbol
    64 64 128     % Fence
    64 0 128      % Car
    64 64 0       % Pedestrian
    0 128 192     % Bicyclist
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;

% Define color labels corresponding to each class
colorLabels = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];
end
