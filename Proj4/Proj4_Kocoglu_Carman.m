% Names: Yildirim Kocoglu & Aaron Carman

clear;
clc;
close all;

%% Load images

disp('Loading images...');

% To generate the same random numbers and reproduce the same results each time
rng(100);

% Trainig data (worm) - Load the reshaped 2-D database (rows = examples, columns = flattened image features)
filepath1 = strcat(pwd,'\TrainingData\','Training_worm.mat');

Training_worm = load(filepath1);
training_worm = Training_worm.original_stored;

% image labels = worm
train_labels_worm = ones(5500,1); % 1 = worm

% Trainig data (noworm) - Load the reshaped 2-D database (rows = examples, columns = flattened image features)
filepath2 = strcat(pwd,'\TrainingData\','Training_noworm.mat');

Training_noworm = load(filepath2);
training_noworm = Training_noworm.original_stored;

% image labels = noworm
train_labels_noworm = zeros(5500,1); % 0 = no worm


%% Divide the data into training, validation, and test sets

disp('Dividing data into training, validation, and test sets...');

% Number of worms = Number of non-worms (5500)
Number_of_images = size(training_worm,1);

% Randomly permutate (shuffle) the images to remove any bias if it exists
idx1 = randperm(Number_of_images);
idx2 = randperm(Number_of_images);

Images_worms = training_worm(idx1,:);
Images_nonworms = training_noworm(idx2,:);

% Total should be equal to 1 (100%)
Percent_training = 0.6;
Percent_validation = 0.2;
Percent_test = 0.2;

% Number of training, validation, test images 
Training_number = round(Number_of_images*Percent_training);
Validation_number = round(Number_of_images*Percent_validation);
Test_number = round(Number_of_images*Percent_test);
% Training, validation, test images - indices
Training_idx = 1:Training_number;
Validation_idx = (Training_number+1):(Training_number+Validation_number);
Test_idx = (Training_number + Validation_number +1):(Training_number+Validation_number+Test_number);



% Divide the images into training, validation, and test data sets

Training_images_worm = Images_worms(Training_idx,:);
Training_images_nonworm = Images_nonworms(Training_idx,:);

Validation_images_worm = Images_worms(Validation_idx,:);
Validation_images_nonworm = Images_nonworms(Validation_idx,:);

Test_images_worm = Images_worms(Test_idx,:);
Test_images_nonworm = Images_nonworms(Test_idx,:);

% Divide the labels into training, validation, and test data sets

Training_labels_worm = train_labels_worm(Training_idx,:);
Training_labels_nonworm = train_labels_noworm(Training_idx,:);

Validation_labels_worm = train_labels_worm(Validation_idx,:);
Validation_labels_nonworm = train_labels_noworm(Validation_idx,:);

Test_labels_worm = train_labels_worm(Test_idx,:);
Test_labels_nonworm = train_labels_noworm(Test_idx,:);

%% Combine all the training, validation, test images and labels

Training_images = [Training_images_worm;Training_images_nonworm];
Validation_images = [Validation_images_worm;Validation_images_nonworm];
Test_images = [Test_images_worm;Test_images_nonworm];

Training_labels = [Training_labels_worm;Training_labels_nonworm];
Validation_labels = [Validation_labels_worm;Validation_labels_nonworm];
Test_labels = [Test_labels_worm;Test_labels_nonworm];

%% SVM algorithm

disp('SVM training in progress...');

C = 10;

tstart = tic;
svm = fitcsvm(Training_images,Training_labels,'Standardize',true,'KernelFunction','rbf','KernelScale','auto','BoxConstraint',C,'Solver','SMO'); % changed from linear to rbf (keep rbf), standardize = true if original data and standardize = false if pca data
tend = toc(tstart);

% Print training time

fprintf('SVM training time is: %0.2f seconds\n', tend);
%% Predict labels using trained svm model - Train

fprintf('\n');
disp('SVM prediction in progress...');

tbegin = tic;
[label_train,~] = predict(svm,Training_images);
tfinish = toc(tbegin);

% Print training prediction time
fprintf('SVM training prediction time is: %0.2f seconds\n', tfinish);
%% Calculate accuracy - Train

[misclassified_index_train,~] = find(label_train ~=Training_labels);

Accuracy_train = 100 - ((length(misclassified_index_train))/length(Training_labels)*100);

% Print training accuracy
fprintf('SVM training accuracy is: %0.2f percent\n', Accuracy_train);
%% Predict labels using trained svm model - Validation

tbegin = tic;
[label_validation,~] = predict(svm,Validation_images);
tfinish = toc(tbegin);

% Print training prediction time
fprintf('SVM validation prediction time is: %0.2f seconds\n', tfinish);
%% Calculate accuracy - Validation

[misclassified_index_validation,~] = find(label_validation ~=Validation_labels);

Accuracy_validation = 100 - ((length(misclassified_index_validation))/length(Validation_labels)*100);

% Print training accuracy
fprintf('SVM validation accuracy is: %0.2f percent\n', Accuracy_validation);
%% Predict labels using trained svm model - Test

tbegin = tic;
[label_test,~] = predict(svm,Test_images);
tfinish = toc(tbegin);

% Print testing time
fprintf('SVM test prediction time is: %0.2f seconds\n', tfinish);
%% Calculate accuracy - Test

[misclassified_index_test,~] = find(label_test ~=Test_labels);

Accuracy_test = 100 - ((length(misclassified_index_test))/length(Test_labels)*100);

% Print testing accuracy
fprintf('SVM test accuracy is: %0.2f percent\n', Accuracy_test);

%% Confusion matrix (chart)

fprintf('\n');
disp('Plotting the Confusion matrix for test images...');

Confusion_matrix_test = confusionmat(Test_labels,label_test); % Test_labels = true labels, label_test = predicted labels
cm = confusionchart(Test_labels,label_test);
cm.Title = strcat('Confusion Matrix for Test Data; Accuracy:', compose('%0.2f',Accuracy_test), '%');
%% Re-train using all of the data and optimum SVM parameters

fprintf('\n');
disp('Re-training using all of the available data...')

% Full training data with all the images (worm + no worm)
Training_data_full = [training_worm;training_noworm];

% Full training labels
Training_labels_full = [train_labels_worm;train_labels_noworm]; 


% SVM training
C = 10; % Changed from 8 to 10
sigma = 67.727905843453970;

tstart = tic;
svm_full = fitcsvm(Training_data_full,Training_labels_full,'Standardize',true,'KernelFunction','rbf','KernelScale',sigma,'BoxConstraint',C,'Solver','SMO'); % changed from linear to rbf (keep rbf), standardize = true if original data and standardize = false if pca data
tend = toc(tstart);

% Print training time
fprintf('SVM re-training time using all the data is: %0.2f seconds\n', tend);

fprintf('\n');
disp('SVM prediction in progress...');
% Predict all the images (training data with all the images)
tbegin = tic;
[label_full,~] = predict(svm_full,Training_data_full);
tfinish = toc(tbegin);

% Print testing time
fprintf('SVM prediction time for all the images (11000 images) is: %0.2f seconds\n', tfinish);


% Calculate accuracy - all the images (11000 images)
[misclassified_index_full,~] = find(label_full ~=Training_labels_full);

Accuracy_full = 100 - ((length(misclassified_index_full))/length(Training_labels_full)*100);

% Print accuracy - all the images (11000 images)
fprintf('SVM accuracy for all the images (11000 images) is: %0.2f percent\n', Accuracy_full);
%fprintf('\nSVM test prediction time is: %0.2f\n', tfinish);

% Predict re-trained model on test images
tbegin = tic;
[label_test_full,~] = predict(svm_full,Test_images);
tfinish = toc(tbegin);

% Print testing time
fprintf('SVM test prediction time with the re-trained model is: %0.2f seconds\n', tfinish);


[misclassified_index_test_full,~] = find(label_test_full ~=Test_labels);

Accuracy_test_full = 100 - ((length(misclassified_index_test_full))/length(Test_labels)*100);

% Print testing accuracy
fprintf('SVM test accuracy with the re-trained model is: %0.2f percent\n', Accuracy_test_full);

%% Save SVM Models (One trained with original size images, one trained after appyling PCA)

fprintf('\n');
disp('Saving the re-trained SVM model...');

saveLearnerForCoder(svm_full, 'SVM'); % SVM model trained with original size images