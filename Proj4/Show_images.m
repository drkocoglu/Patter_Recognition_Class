clear;
clc;
close all;

% %% Training check
% Reshaped_image = reshape(Reduced_Training_images(1,:),[70,70]);
% 
% figure()
% imshow(Reshaped_image)
% 
% figure()
% imshow(reshape(Training_images(1,:),[101,101]))
% 
% A = Reduced_Training_images*coeff(:,1:4900)';
% 
% figure()
% imshow(reshape(A(1,:),[101,101]))
% 
% %% Test check
% 
% Reshaped_image = reshape(Reduced_Test_images(1,:),[70,70]);
% 
% figure()
% imshow(Reshaped_image)
% 
% figure()
% imshow(reshape(Test_images(1200,:),[101,101]))
% 
% A = Reduced_Test_images*coeff(:,1:4900)' + mu;
% 
% figure()
% imshow(reshape(A(1200,:),[101,101]))
% 
% %% score check (using score to train and for test image (test_images -mu)*coeff(:,1:idx) gives similar images after reconstruction
% 
% Reshaped_image = reshape(score(1,1:4900),[70,70]);
% 
% figure()
% imshow(Reshaped_image)
% 
% figure()
% imshow(reshape(Training_images(1,:),[101,101]))
% 
% A = score(:,1:4900)*coeff(:,1:4900)' + mu;
% 
% figure()
% imshow(reshape(A(1,:),[101,101]))

Directory = 'C:\Users\ykocoglu\Desktop\PATTERN RECOGNITION\Assignments\Proj4\Celegans_ModelGen\1'; 
% Read images from Images folder
Imgs = dir(strcat(Directory,'\','*.png'));
info = imfinfo(strcat(Directory,'\',Imgs(3).name));
height = info.Height
width = info.Width
number_of_images = numel(Imgs)
% Image_names = struct2table(Imgs);
% sortedT = sortrows(Image_names);
% for j=1:length(Imgs)
%     thisname = Imgs(j).name;
%     thisfile = fullfile(Directory, thisname);
%     try
%       Img = imread(thisfile);  % try to read image
%       Im = Img(:,:,1);
%       %figure    
%       %imshow(Im)
%       title(thisname);
%    catch
%    end
% end