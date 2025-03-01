% Names: Yildirim Kocoglu & Aaron Carman

clear;
clc;
close all;

%% Import images using the function read_images

% Image type and image names
image_type = '.png';
image_name = 'image_';

disp('Please modify the location path of the images in location1 and location2 variables if necessary...');

% Location of worm images
location1 = 'C:\Users\ykocoglu\Desktop\PATTERN RECOGNITION\Assignments\Proj4\Celegans_ModelGen\1';
% Location of non-worm (defect) images
location2 = 'C:\Users\ykocoglu\Desktop\PATTERN RECOGNITION\Assignments\Proj4\Celegans_ModelGen\0';

% Names of the ".mat" files to be saved into the "TrainingData" Folder created during the use of read_images() function
Mat_file_name1 = 'Training_worm.mat';
Mat_file_name2 = 'Training_noworm.mat';

% Read the images and save it as flattened images to a new folder called TrainingData in its original size
read_images(location1,image_name,image_type,Mat_file_name1);
read_images(location2,image_name,image_type,Mat_file_name2);


%% Function to import the flattened "original" size images
function read_images(location,image_name,image_type,Mat_file_name)

filepath = strcat(location,'\',image_name,num2str(1),image_type);
info = imfinfo(filepath);
Height = info.Height;

% size of original flattened image
flattened_size = Height*Height;

% Create variables to store all flattened images
original_stored = zeros(1,flattened_size);


% Count number of images in the BoxedImages
a = strcat('\*',image_type) ; % image type (jpeg, png, etc.)
number_of_images = numel(dir(fullfile(location,a))); % count number of images in the specified location (folder)

% Throw an error if there are no images inside the folder
if number_of_images == 0
    error('There are no images inside the specified folder!\n')
end

%%% Import the images in a for loop and report time it takes in seconds
tic;
for i = 1:number_of_images
    
    filepath = strcat(location,'\',image_name,num2str(i),image_type);
    
    % Original Data
    original = imread(filepath); % read the original image
    original = imbinarize(original); % binarize image
    original_converted = im2double(original); % convert image to double
    original_reshaped = reshape(original_converted,[1,flattened_size]); % flatten the image
    
    % Store the reshaped image
    original_stored(i,:) = original_reshaped;
    
end
toc;

%%% Save flattened images as a mat file for later use (Training and Test Data combined)

% Check if TrainingData folder exists in the current directory and if not create it
if ~exist('TrainingData', 'dir')
    mkdir TrainingData;
end

% Mat_file_name = 'Training.mat';
filepath = strcat(pwd,'\TrainingData\',Mat_file_name);

save(filepath,'original_stored');

end