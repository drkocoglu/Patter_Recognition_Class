% Assignment#3 (Pattern Recognition)
% Name: Yildirim Kocoglu

clear;
clc;
close all;

%% Import Data

% True means for each class
m = [zeros(5,1), ones(5,1)];

% True covariance matrix for each class
S(:,:,1)=[0.8 0.2 0.1 0.05 0.01; 0.2 0.7 0.1 0.03 0.02; 0.1 0.1 0.8 0.02 0.01; 0.05 0.03 0.02 0.9 0.01; 0.01 0.02 0.01 0.01 0.8];
S(:,:,2)=[0.9 0.1 0.05 0.02 0.01; 0.1 0.8 0.1 0.02 0.02; 0.05 0.1 0.7 0.02 0.01; 0.02 0.02 0.02 0.6 0.02; 0.01 0.02 0.01 0.02 0.7];

% Prior probability for each class (P(Ci))
P=[1/2 1/2]';

% 100 and 1000 training samples
N_train = [100,1000];

% 10000 test samples
N_test = 10000;

rng(0);
[X_train_1,Y_train_1] = generate_gauss_classes(m,S,P,N_train(1));

rng(0);
[X_train_2,Y_train_2] = generate_gauss_classes(m,S,P,N_train(2));

rng(100);
[X_test_1,Y_test_1] = generate_gauss_classes(m,S,P,N_test);

%% MLE for Naive Bayes classifier (N=100 and N=1000)

% MLE for parameters for Naive Bayes (N=100)
[m_ML_train_1_naive,S_ML_train_1_naive] = ML_estimate_naive(X_train_1,Y_train_1);

% MLE for parameters for Naive Bayes (N=1000)
[m_ML_train_2_naive,S_ML_train_2_naive] = ML_estimate_naive(X_train_2,Y_train_2);

%% MLE for Bayes classifier (N=100 and N=1000)

% MLE for Bayes classifier Xtrain1 (N=100)
[m_ML_train_1,S_ML_train_1] = ML_estimate(X_train_1,Y_train_1);

% MLE for Bayes classifier Xtrain2 (N=1000)
[m_ML_train_2,S_ML_train_2] = ML_estimate(X_train_2,Y_train_2);

%% i) Naive Bayes Classifier Predictions for both training and test data

prediction_naive_train1 = Naive_Bayes_classifier(P,m_ML_train_1_naive,S_ML_train_1_naive,X_train_1);
prediction_naive_train2 = Naive_Bayes_classifier(P,m_ML_train_2_naive,S_ML_train_2_naive,X_train_2);

prediction_naive_test1 = Naive_Bayes_classifier(P,m_ML_train_1_naive,S_ML_train_1_naive,X_test_1);
prediction_naive_test2 = Naive_Bayes_classifier(P,m_ML_train_2_naive,S_ML_train_2_naive,X_test_1);

%% ii) Bayes Classifier with ML estimate for both training and test data
prediction_ML_train1 = Bayes_classifier(P,m_ML_train_1,S_ML_train_1,X_train_1);
prediction_ML_train2 = Bayes_classifier(P,m_ML_train_2,S_ML_train_2,X_train_2);

prediction_ML_test1 = Bayes_classifier(P,m_ML_train_1,S_ML_train_1,X_test_1);
prediction_ML_test2 = Bayes_classifier(P,m_ML_train_2,S_ML_train_2,X_test_1);

%% iii) Bayes Classifier with true parameters for both training and test data
prediction_train1_true = Bayes_classifier(P,m,S,X_train_1);
prediction_train2_true = Bayes_classifier(P,m,S,X_train_2);
prediction_test_true = Bayes_classifier(P,m,S,X_test_1);

%% Calculate Test Accuracy and Test Error for all classifiers

% Bayes classifier with true parameters
[Accuracy_true,Error_true] = Calculate_Accuracy_error(prediction_test_true,Y_test_1);

% Bayes classifier with ML estimate from Xtrain (N=100 and N=1000)
[Accuracy_1_ML,Error_1_ML] = Calculate_Accuracy_error(prediction_ML_test1,Y_test_1);
[Accuracy_2_ML,Error_2_ML] = Calculate_Accuracy_error(prediction_ML_test2,Y_test_1);

% Naive Bayes classifier with ML estimate from Xtrain (N=100 and N=1000)
[Accuracy_1_naive,Error_1_naive] = Calculate_Accuracy_error(prediction_naive_test1,Y_test_1);
[Accuracy_2_naive,Error_2_naive] = Calculate_Accuracy_error(prediction_naive_test2,Y_test_1);

%% Show Test Accuracy and Test Errors for all classifiers in a table

% Show the Test Accuracy and Test Errors for each classifier (Naive Bayes, Bayes with ML estimate, Bayes with true parameters) in a table

Total_Accuracy = [Accuracy_true;Accuracy_1_naive;Accuracy_1_ML;Accuracy_2_naive;Accuracy_2_ML];
Total_Error = [Error_true;Error_1_naive;Error_1_ML;Error_2_naive;Error_2_ML];
Mytable = [Total_Accuracy,Total_Error];

T = array2table(Mytable,'VariableNames',{'Test Accuracy','Test Error'});

T.Properties.RowNames = {'Bayes with true parameters','Naive Bayes (N = 100)','Bayes with MLE (N = 100)','Naive Bayes (N = 1000)','Bayes with MLE (N = 1000)'};

fig = uifigure;
uit = uitable(fig,'Data',T, 'Position',[30 150 510 161]);



%% Function to generate classes from the gaussian distribution


function [X,y]=generate_gauss_classes(m,S,P,N)

[~,c]=size(m);
X=[];
y=[];

for j=1:c
    % Generating the [p(j)*N)] vectors from each distribution
    % Takes P(j)*N points to randomly generate multivariate normal samples using the mean m and standard deviation S.
    t=mvnrnd(m(:,j),S(:,:,j),fix(P(j)*N))'; % The total number of points may be slightly less than N due to the fix operator.
    X=[X, t];
    y=[y, ones(1,fix(P(j)*N))*j];
end

end

%% Function for Bayes classifier

function prediction = Bayes_classifier(P,m,S,X)

% Initialize parameters needed for calculating posterior probability of each sample for each class
[N,d] = size(X');
[~,c] = size(m);
posterior_prob = zeros(c,1);
prediction = zeros(N,1);

% Define the conditional probability - P(X|Ci) function for multivariate normal pdf
Cond_prob = @(d,m,S,X) (1./(((2.*pi).^(d/2)).*det(S).^0.5)).*exp((-1/2).*(X-m)'*inv(S)*(X-m));

% Calculate posterior probability of each sample for each class and predict the classes
for j = 1:N
    for i =1:c
        posterior_prob(i,:) = P(i).*Cond_prob(d,m(:,i),S(:,:,i),X(:,j)); % Calculate posterior probability of each class
    end
    [~,prediction(j)] = max(posterior_prob); % Calculate max posterior probability for each sample and find the index and store (index = class#)
end

end
%% Function for Naive Bayes classifier

function prediction = Naive_Bayes_classifier(P,m,S,X)

% Define the conditional probability - P(X|Ci) function for multivariate normal pdf
Cond_prob_naive = @(m,S,X) (1./(((2.*pi).*S).^0.5)).*exp((-1/(2.*S)).*(X-m).^2);

% Initialize parameters needed for calculating posterior probability of each sample for each class
[N,d] = size(X');
[~,c] = size(m);
Cond_prob_matrix = zeros(d,N);
Cond_prob_cell = cell(c,1);
posterior_prob_naive = zeros(c,N);

for j = 1:c
    for i=1:d
        Cond_prob_matrix(i,:)=Cond_prob_naive(m(i,j),S(i,j),X(i,:)); % Conditional probability for class 1 & 2 for each feature
    end
    Cond_prob_cell{j,1} = Cond_prob_matrix;
    posterior_prob_naive(j,:) = P(j)*prod(Cond_prob_cell{j,1}); % Calculate posterior probability
end


% Prediction
[~,prediction] = max(posterior_prob_naive);

prediction = prediction';

end
%% Function for ML estimate for Naive Bayes classifier

function [m_ML_naive, S_ML_naive] = ML_estimate_naive(X,y)

[~,d] = size(X');

class_labels = unique(y);
number_of_classes = length(class_labels);

S_ML_naive=zeros(d,number_of_classes);
m_ML_naive = zeros(d,number_of_classes);

for i=1:number_of_classes
    
    Total = length(X(:,y==class_labels(i))); % Total number of samples in each class (Depends on P(Ci) - prior probability used for generating the data)
    
    m_ML_naive(:,i) = (1/Total)*(sum(X(:,y==class_labels(i))'))'; 
    
    % Independent features
    for j = 1:d
        S_ML_naive(j,i) = (1/Total)*(X(j,y==class_labels(i))-m_ML_naive(j,i))*(X(j,y==class_labels(i))-m_ML_naive(j,i))';
    end
    
end

end

%% Function for ML estimate for Bayes classifier

function [m_ML,S_ML] = ML_estimate(X,y)

[~,d] = size(X');
class_labels = unique(y);
number_of_classes = length(class_labels);
m_ML = zeros(d,number_of_classes);
S_ML = zeros(d,d,number_of_classes);

% Dependent features
for i=1:number_of_classes
    
    Total = length(X(:,y==class_labels(i))); % Total number of samples in each class (Depends on P(Ci) - prior probability used for generating the data)
    
    m_ML(:,i) = (1/Total)*(sum(X(:,y==class_labels(i))'))'; % Each class has half the number of samples therefore, 2/N gives total samples for each class
    S_ML(:,:,i) = (1/Total).*(X(:,y==class_labels(i))-m_ML(:,i))*(X(:,y==class_labels(i))-m_ML(:,i))'; % Each class has half the number of samples therefore, 2/N gives total samples for each class
end

end

%% Function for calculating Test Error and Accuracy

function [Accuracy,Error] = Calculate_Accuracy_error(prediction,y)

% Calculating Test Error and Accuracy (in percent)
Error = (sum(y'~=prediction)./length(prediction))*100;
Accuracy = 100-(Error);

end