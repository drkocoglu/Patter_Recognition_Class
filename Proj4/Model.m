
% Names: Yildirim Kocoglu & Aaron Carman

clear;
clc;
close all;

%% Load images
disp('WARNING: This code may take some time to run...');
fprintf('\n');
disp('Please change image type if necessary (options: .png, ,.jpeg, etc.)...')
% Image type
image_type = '.png';

% Location of the images to be predicted
fprintf('\n');
location1 = input('Please enter the location of the images to be predicted...\n', 's');

fprintf('\n');
disp('Loading images...');
% Load the images (test_images) and image names (image_store)
[test_images, image_store] = read_images(location1,image_type);

%% Load the models
fprintf('\n');
disp('Loading the saved SVM model...');
SVM = loadLearnerForCoder('SVM.mat');

%% Predict labels using SVM model (original size images)
fprintf('\n');
disp('Prediction in progress...');
[label_test,~] = predict(SVM,test_images);

%% Store labels
disp('Prediction completed. Please look at the "Prediction_list" cell array saved in workspace of MATLAB to see the name & predicted labels of the images.');
Predictions_list = [image_store(:,1),num2cell(label_test)];

%% Number of worms (1) and nonworms (0)

fprintf('\n');
disp('Counting the number of worms(1) and nonworms(0)...');

[index_1, ~] = find(label_test == 1); % Worms
[index_0, ~] = find(label_test == 0); % Non-worms

number_of_worms = length(index_1);
number_of_nonworms = length(index_0);

fprintf('Number of worms is: %0.f\n', number_of_worms);
fprintf('Number of nonworms is: %0.f\n', number_of_nonworms);

%% Aasfsa
predicted_items = cell(2,2);

predicted_items{1,1} = 'Number of worms';
predicted_items{2,1} = 'Number of no worms';
predicted_items{1,2} = number_of_worms;
predicted_items{2,2} = number_of_nonworms;

Predictions_list = [Predictions_list; predicted_items];

%% Function to import the flattened "original" size images
function [original_stored, image_store] = read_images(location,image_type)

% Get the names of all images in the specified folder
images = dir(strcat(location,'\*',image_type)) ; % image type (jpeg, png, etc.)

% Number of images in the specified folder
number_of_images = numel(images);

% Throw an error if there are no images inside the folder
if number_of_images == 0
    error('There are no images inside the specified folder!\n')
end

% Get Height and Width of the images in the specified folder
info = imfinfo(strcat(location,'\',images(1).name));
Height = info.Height;
Width = info.Width;

% size of original flattened image
flattened_size = Height*Width;

% Create variables to store all flattened images
original_stored = zeros(1,flattened_size);

% Initialize two column list for "image_name + labels"
image_store = cell(number_of_images,1);



%%% Import the images in a for loop and report time it takes in seconds
tbegin = tic;
for i = 1:number_of_images
    
    filepath = strcat(location,'\',images(i).name);
    
    % Two column list for "image names + labels"
    image_store{i,1} = images(i).name;
    
    % Original Data
    original = imread(filepath); % read the original image
    original = imbinarize(original); % binarize image
    original_converted = im2double(original); % convert image to double
    original_reshaped = reshape(original_converted,[1,flattened_size]); % flatten the image
    
    % Store the reshaped image
    original_stored(i,:) = original_reshaped;
    
end
tfinish = toc(tbegin);

% Print testing time
fprintf('Loading time is: %0.2f seconds\n', tfinish);

end