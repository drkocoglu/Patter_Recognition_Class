% Assignment_1 (Pattern Recognition)
% Name: Yildirim Kocoglu

% Please run the code one section at a time to see the results in workspace!
% Some variables such as w,w2,etc. are repeated and replaced to avoid giving different & long names to each variable

%% Clear and close all
clear;
clc;
close all;
all_fig = findall(0, 'type', 'figure');
close(all_fig)
%% Load fisheriris data
load fisheriris.mat;

%% Set the random number generator seed
rng(100, 'twister');
%% Create plots to visualize the fisheriris data

% Plot sepal length vs sepal width for each class

% Sepal length for each flower type
X1_setosa = meas(1:50, 1);
X1_versicolor = meas(51:100,1);
X1_virginica = meas(101:150,1); 

% Sepal width for each flower type
X2_setosa = meas(1:50, 2);
X2_versicolor = meas(51:100,2);
X2_virginica = meas(101:150,2); 

figure();
p1 = scatter(X1_setosa, X2_setosa, 'o', 'r');
hold on
p2 = scatter(X1_versicolor, X2_versicolor, 'd', 'b');
p3 = scatter(X1_virginica, X2_virginica, 'x', 'g');
hold off
title('Sepal length vs Sepal width');
xlabel('Sepal Length');
ylabel('Sepal width');
legend([p1,p2,p3],{'setosa','versicolor','virginica'}, 'Location','southeast');

% Plot petal length vs petal width for each class

% Petal length for each flower type
X3_setosa = meas(1:50, 3);
X3_versicolor = meas(51:100,3);
X3_virginica = meas(101:150,3); 

% Petal width for each flower type
X4_setosa = meas(1:50, 4);
X4_versicolor = meas(51:100,4);
X4_virginica = meas(101:150,4);

figure();
p1 = scatter(X3_setosa, X4_setosa, 'o', 'r');
hold on
p2 = scatter(X3_versicolor, X4_versicolor, 'd', 'b');
p3 = scatter(X3_virginica, X4_virginica, 'x', 'g');
hold off
title('Petal length vs Petal width');
xlabel('Petal Length');
ylabel('Petal width');
legend([p1,p2,p3],{'setosa','versicolor','virginica'}, 'Location','southeast');

%% Compute the following quantities for each feature. Do you observe anything of interest from these statistics?

min_meas = [min(meas(:,1)), min(meas(:,2)), min(meas(:,3)), min(meas(:,4))];
max_meas = [max(meas(:,1)), max(meas(:,2)), max(meas(:,3)), max(meas(:,4))];
mean_meas = [mean(meas(:,1)), mean(meas(:,2)), mean(meas(:,3)), mean(meas(:,4))];
var_meas = [var(meas(:,1)), var(meas(:,2)), var(meas(:,3)), var(meas(:,4))];

% Prior probability of each class
Pj = 50/150;

% Within-class Variance
sw = [Pj.*(var(X1_setosa) + var(X1_versicolor) + var(X1_virginica)), Pj.*(var(X2_setosa) + var(X2_versicolor) + var(X2_virginica)), Pj.*(var(X3_setosa) + var(X3_versicolor) + var(X3_virginica)), Pj.*(var(X4_setosa) + var(X4_versicolor) + var(X4_virginica))];


class_mean_setosa = [mean(X1_setosa), mean(X2_setosa), mean(X3_setosa), mean(X4_setosa)];
class_mean_versicolor = [mean(X1_versicolor), mean(X2_versicolor), mean(X3_versicolor), mean(X4_versicolor)];
class_mean_virginica = [mean(X1_virginica), mean(X2_virginica), mean(X3_virginica), mean(X4_virginica)];

between_class_variance_setosa = Pj.*(class_mean_setosa - mean_meas).^2;
between_class_variance_versicolor = Pj.*(class_mean_versicolor - mean_meas).^2;
between_class_variance_virginica = Pj.*(class_mean_virginica - mean_meas).^2;

% Between-class Variance
sb = between_class_variance_setosa + between_class_variance_versicolor + between_class_variance_virginica;

% Show the statistics in a table
Mytable = [min_meas; max_meas; mean_meas; var_meas; sw; sb];

T = array2table(Mytable,...
    'VariableNames',{'Sepal length','Sepal width','Petal lenth', 'Petal width'});

T.Properties.RowNames = {'Minimum','Maximum','Mean','Variance','Within-Class variance', 'Between-Class variance'};

fig = uifigure;
uit = uitable(fig,'Data',T, 'Position',[30 150 510 161]);

%% Correlation coefficient plot

Class_vector = [ones(50,1); 2.*ones(50,1); 3.*ones(50,1)];
new_matrix = [meas, Class_vector];

R = corrcoef(new_matrix);

figure()
imagesc(R);
set(gca, 'XTick', 1:5); % center x-axis ticks on bins
set(gca, 'YTick', 1:5); % center y-axis ticks on bins
set(gca, 'XTickLabel', {'SepL', 'SepW', 'PetL','PetW','Class'}); % set x-axis labels
set(gca, 'YTickLabel', {'SepL', 'SepW', 'PetL','PetW','Class'}); % set y-axis labels
colormap('jet');
colorbar;
caxis([-0.4 1]);

%% Display each of four features versus the class label. What can you state about how well the features may perform in classification?

figure()
t = tiledlayout(2,2);
nexttile
scatter(meas(:,1),Class_vector, 'x', 'r')
title('SepL vs Class');
xlim([0 8]);
nexttile
scatter(meas(:,2),Class_vector, 'x', 'r')
title('SepW vs Class');
xlim([0 8]);
nexttile
scatter(meas(:,3),Class_vector, 'x', 'r')
title('PetL vs Class');
xlim([0 8]);
nexttile
scatter(meas(:,4),Class_vector, 'x', 'r')
title('PetW vs Class');
xlim([0 8]);

%% Set initial data for Least squares and Batch Perceptron

% Use as input data for least squares method
All_features  = [meas,ones(size(meas,1),1)];
Partial_features = [All_features(:,3:4),ones(size(meas,1),1)];
class = [1, 0];

% Use as initial w for batch perceptron
w_initial = rand(size(All_features,2),1); % initial w for all features
w_initial_partial = rand(size(Partial_features,2),1); % initial w for partial features

%% All Features (Setosa vs. Versi + Virgi)

for i = 1:70
fprintf('-');
end

fprintf('\nAll features with Setosa vs. Versi + Virgi\n\n');

% Use as input data for batch perceptron method
Data = All_features;
Data(51:150,:) = -1.*Data(51:150,:); % multiplying class 2 with -1

% Use as labels for both least squares and batch perceptron
labels = [ones(50,1);zeros(100,1)];


% Hyperparameters for batch perceptron
rho = 0.1;
max_epochs = 1000;


[w, prediction_LS] = least_squares(All_features, labels, class);
[w2, iteration_number, prediction_perceptron] = batch_perceptron(Data, labels, rho, w_initial, max_epochs);

% Number of misclassifications for LS and BP
misclassified_LS = length(prediction_LS(prediction_LS == -1));
misclassified_perceptron = length(prediction_perceptron(prediction_perceptron == -1));
%% Features 3 and 4 only (Setosa vs. Versi + Virigi)

for i = 1:70
fprintf('-');
end

fprintf('\nFeatures 3 & 4 only features with Setosa vs. Versi + Virgi\n\n');

% Use as input data for batch perceptron method
Data = Partial_features;
Data(51:150,:) = -1.*Data(51:150,:); % multiplying class 2 with -1

% Use as labels for both least squares and batch perceptron
labels = [ones(50,1);zeros(100,1)];


% Hyperparameters for batch perceptron
rho = 0.1;
max_epochs = 1000;


[w, prediction_LS] = least_squares(Partial_features, labels, class);
[w2, iteration_number, prediction_perceptron] = batch_perceptron(Data, labels, rho, w_initial_partial, max_epochs);

% Number of misclassifications for LS and BP
misclassified_LS = length(prediction_LS(prediction_LS == -1));
misclassified_perceptron = length(prediction_perceptron(prediction_perceptron == -1));

% Plot the decision boundary

X1 = (-(w2(3) + Partial_features(:,1)*w2(1))/w2(2)); % Batch Perceptron Decision Boundary
X2 = (-(-0.5 + w(3) + Partial_features(:,1)*w(1))/w(2)); % Least Squares Decision Boundary

figure();
p1 = scatter(X3_setosa, X4_setosa, 'o', 'r');
hold on
p2 = scatter(X3_versicolor, X4_versicolor, 'd', 'b');
p3 = scatter(X3_virginica, X4_virginica, 'x', 'g');
p4 = plot(meas(:,3), X2);
p5 = plot(meas(:,3), X1);
hold off
title('Setosa vs. Versi + Virigi');
xlabel('Petal Length');
ylabel('Petal width');
ylim([0 2.5]);
legend([p1,p2,p3,p4],{'setosa','versicolor','virginica','Decision boundary'},'Location','southeast');
legend([p1,p2,p3,p4,p5],{'setosa','versicolor','virginica','LS - Decision Boundary', 'BP - Decision Boundary'},'Location','southeast');

%% All Features (Virgi vs. Versi + Setosa)

for i = 1:70
fprintf('-');
end

fprintf('\nAll features with Virgi vs. Versi + Setosa\n\n');

% Use as input data for batch perceptron method
Data = All_features;
Data(1:100,:) = -1.*Data(1:100,:); % multiplying class 2 with -1

% Use as labels for both least squares and batch perceptron
labels = [ones(100,1);zeros(50,1)];


% Hyperparameters for batch perceptron
rho = 0.1;
max_epochs = 1000;


[w, prediction_LS] = least_squares(All_features, labels, class);
[w2, iteration_number, prediction_perceptron] = batch_perceptron(Data, labels, rho, w_initial, max_epochs);

% Number of misclassifications for LS and BP
misclassified_LS = length(prediction_LS(prediction_LS == -1));
misclassified_perceptron = length(prediction_perceptron(prediction_perceptron == -1));

%% Features 3 and 4 only (Virgi vs. Versi + Setosa)

for i = 1:70
fprintf('-');
end

fprintf('\nFeatures 3 & 4 only with Virgi vs. Versi + Setosa\n\n');

% Use as input data for batch perceptron method
Data = Partial_features;
Data(1:100,:) = -1.*Data(1:100,:); % multiplying class 2 with -1

% Use as labels for both least squares and batch perceptron
labels = [ones(100,1);zeros(50,1)];


% Hyperparameters for batch perceptron
rho = 0.1;
max_epochs = 1000;


[w, prediction_LS] = least_squares(Partial_features, labels, class);
[w2, iteration_number, prediction_perceptron] = batch_perceptron(Data, labels, rho, w_initial_partial, max_epochs);


% Number of misclassifications for LS and BP
misclassified_LS = length(prediction_LS(prediction_LS == -1));
misclassified_perceptron = length(prediction_perceptron(prediction_perceptron == -1));

% Plot the decision boundary

X1 = (-(w2(3) + Partial_features(:,1)*w2(1))/w2(2)); % Batch Perceptron Decision Boundary
X2 = (-(-0.5 + w(3) + Partial_features(:,1)*w(1))/w(2)); % Least Squares Decision Boundary

figure();
p1 = scatter(X3_setosa, X4_setosa, 'o', 'r');
hold on
p2 = scatter(X3_versicolor, X4_versicolor, 'd', 'b');
p3 = scatter(X3_virginica, X4_virginica, 'x', 'g');
p4 = plot(meas(:,3), X2);
p5 = plot(meas(:,3), X1);
hold off
title('Virgi vs Versi + Setosa');
xlabel('Petal Length');
ylabel('Petal width');
ylim([0 2.5]);
legend([p1,p2,p3,p4,p5],{'setosa','versicolor','virginica','LS - Decision Boundary', 'BP - Decision Boundary'},'Location','southeast');


%% Multi-Class Least Squares (Setosa vs Versi vs Virgi) using Features 3 & 4 only

Data = Partial_features;
classes = eye(3,3); % Create the class labels for each class
% Labels using one-hot encoding
labels = [repmat(classes(1,:),50,1);repmat(classes(2,:),50,1);repmat(classes(3,:),50,1)];
% Create target labels as 1,2,3
target = [ones(50,1); 2.*ones(50,1); 3.*ones(50,1)];

for i = 1:70
fprintf('-');
end

fprintf('\nFeatures 3 & 4 only with Virgi vs. Versi + Setosa (Multi-class LS)\n\n');

fprintf('\nApplying Multi-class Least Squares!\n');

% Find W matrix using Multi-class LS
W = pinv(Data'*Data)*(Data'*labels);
% Predict labels using W from Multi-class LS
prediction = Data*W;

% Find the maximum column number (possible answers --> 1,2, or 3) = prediction
[A,colindex] = max(prediction,[],2);

% Find the indices where target labels don't match the predicted labels (misclassified)
[index, ~] = find(target ~= colindex);

% Number of misclassifications for Multi-class LS
misclassified_MultiLS = length(index);


fprintf('Multi-class Least Squares has been applied!\n');

% Decision functions
w12 = W(:,1)-W(:,2); % d1-d2 (w1-w2)
w23 = W(:,2)-W(:,3); % d2-d3 (w2-w3)
w31 = W(:,3)-W(:,1); % d3-d1 (w3-w1)

% Decision boundaries for Multi-class LS
X1 = (-(w12(3) + Partial_features(:,1)*w12(1))/w12(2));  % Multi-class Least Squares Decision Boundary
X2 = (-(w23(3) + Partial_features(:,1)*w23(1))/w23(2)); % Multi-class Least Squares Decision Boundary
X3 = (-(w31(3) + Partial_features(:,1)*w31(1))/w31(2)); % Multi-class Least Squares Decision Boundary

% Plot decision boundaries for Multi-class LS
figure();
p1 = scatter(X3_setosa, X4_setosa, 'o', 'r');
hold on
p2 = scatter(X3_versicolor, X4_versicolor, 'd', 'b');
p3 = scatter(X3_virginica, X4_virginica, 'x', 'g');
p4 = plot(meas(:,3), X1);
p5 = plot(meas(:,3), X2);
p6 = plot(meas(:,3), X3);
hold off
title('Virgi vs Versi + Setosa');
xlabel('Petal Length');
ylabel('Petal width');
ylim([0 2.5]);
legend([p1,p2,p3,p4,p5,p6],{'setosa','versicolor','virginica','Decision Boundary-12', 'Decision Boundary-23', 'Decision Boundary-31'},'Location','southeast');



%% Batch perceptron function

function [w2, iteration_number, predictions] = batch_perceptron(Data, labels, rho, w_initial, max_epochs)

fprintf('\nApplying Batch Perceptron!\n');

iteration = 0;
w2 = w_initial;
tolerance = 1;

while tolerance > 1e-6
    w3 = w2;
    y = Data*w2;
    [index,~] = find(y < 0);
    x = Data(index,:);
    w2 = w2 + rho.*sum(x)';
    tolerance = abs(norm(w3 - w2));
    
    iteration = iteration + 1;
    fprintf('Iteration = %d\n',iteration);
    
    if iteration == max_epochs
        fprintf('Maximum number of Epochs are reached for Batch Perceptron!\n');
        break;
    end
end

iteration_number = iteration;
predictions = y;

[correctly_classified, ~] = find(predictions >= 0);
[misclassified, ~] = find(predictions < 0);

predictions(correctly_classified) = labels(correctly_classified);
predictions(misclassified) = -1; % If prediction = -1, then,  misclassified point

fprintf('Batch Perceptron has been applied!\n');
end

%% Least squares function

function [w, prediction] = least_squares(Data, labels, class)

% class variable must match labels variable

fprintf('Applying Least Squares!\n');

w = pinv(Data'*Data)*(Data'*labels);

prediction = Data*w;
prediction(prediction >= 0.5) = class(1); 
prediction(prediction <0.5) = class(2);

[misclassified, ~] = find(labels ~= prediction);

prediction(misclassified) = -1; % If prediction = -1, then,  misclassified point

fprintf('Least Squares has been applied!\n');

end