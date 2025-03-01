% Assignment_2 (Pattern Recognition)
% Name: Yildirim Kocoglu

clear;
clc;
close all;

% Calculate elapsed time for the whole code to finish with tic-toc;
Begin = tic;

% Generate the required data for class 1 and class 2
rng(100); 
class1=mvnrnd([1 3],[1 0; 0 1],60); 
class2=mvnrnd([4 1],[2 0; 0 2],40);

% Given C values
C = [0.1,100];

%% Part 1

for i = 1:length(C) % for C = 0.1 and C = 100
    
% Combine class features and generate the labels for class 1 and class 2
class_features = [class1;class2];
labels = [ones(length(class1),1);-1.*ones(length(class2),1)];
N = size(class_features,1); % Size of samples

% Prepare data for quadprog by finding H,f (x = lambda)
% Note: quadprog finds the minimum for a quadratic cost function but, we need to maximize the given cost function for the dual form which causes f and H terms to be multiplied by -1.
H = (class_features*class_features').*(labels*labels');
f = -1.*ones(N,1);

% Specify the lower and upper bound
lb = zeros(N,1);
ub = repmat(C(i),N,1); % C = 0.1 or C = 100

% Note: There are no inequality constraints for the dual form besides the lower and upper bounds so, A = [] and b = [] for (A.x <= b) in quadprog
A = [];
b =[];

% Find Aeq and beq (equality constraints) for the constraint Aeq*x = beq (y'*lambda = 0) where lambda = x
Aeq = labels';
beq = 0;

% Set Quadprog for finding lambda
lambda = quadprog(H,f, A, b, Aeq, beq, lb,ub);
lambda(lambda < 1E-5) = 0; % set lambda = 0 for lambda < 1*10^-5
lambda = round(lambda, 4); % round the value to the nearest 4th significant digit for stability of the solution

% Find w (equation from page 126 (section 3) of the Pattern Recogniticon book)
w = (class_features'*(labels.*lambda)); 

% Find the indices of support vectors
[Sup_vec, values] = find(lambda > 0 & lambda < C(i)); 

% Find wo using equation 3.100 in the book (KKT conditions), and only (strictly) using the features and labels associated with support vectors (0<lambda<C) so that slack variables = eps(i) = 0 and can be removed from the equation.
% Note: Using lambda = C leads to mu(i) = 0 from the constraint C - m(i) - lambda(i) = 0 and from the constraint mu(i)*eps(i) = 0 if mu(i) = 0 then eps(i) >= 0 (not necessarily equals zero) where eps(i) are the slack variables
% Note: The point of using this trick above is because wo is expected to be the same value for all the examples in d(x) = wx(i) + wo but, we are left with 2 unknowns in equation 3.100 after finding w (wo and eps(i))
wo = mean((1./labels(Sup_vec)) - class_features(Sup_vec,:)*w);

% Calculate margin length
margin = 2/norm(w);


% Find the boundaries
X2 = -(w(1)*class_features(:,1) + wo)/w(2); % Decision boundary
X3 = (1 -(w(1)*class_features(:,1) + wo))/w(2); % Upper margin
X4 = (-1 -(w(1)*class_features(:,1) + wo))/w(2); % Lower margin

% Find misclassified features
dx = class_features*w + wo;
[misclassified_index,missclass_value] = find(dx(1:length(class1)) < 0); 
misclassified = class_features(misclassified_index,:);
[misclassified_index_2,missclass_value_2] = find(dx(length(class1)+1:length(class_features)) > 0);
misclassified_index_2 = misclassified_index_2 + length(class1);
misclassified_2 = class_features(misclassified_index_2,:);
total_misclassified = [misclassified;misclassified_2];

% Find support vectors
support_vectors = class_features(Sup_vec,:); % Support vectors for 0 < lambda < C

% Find support vectors inside margin
[inside_margin_index, inside_margin_values] = find(lambda == C(i));
inside_margin = class_features(inside_margin_index,:);

% Total number of support vectors (lambda > 0)
total_support_vectors = [support_vectors;inside_margin];

% Plot the decision boundary (wx + wo = 0) and the margins (wx + wo = 1 and wx + wo = -1)
figure(1);
subplot(1,2,i)
scatter(class1(:,1),class1(:,2),200,'o','r','filled','MarkerEdgeColor','k','Linewidth',1);
hold on
scatter(class2(:,1), class2(:,2),200, 's', 'y','filled','MarkerEdgeColor','k','Linewidth',1);
plot(class_features(:,1),X2);
plot(class_features(:,1),X3,'LineWidth',1,'LineStyle',':','Marker','*');
plot(class_features(:,1),X4,'LineWidth',1,'LineStyle',':','Marker','*');
scatter(total_misclassified(:,1),total_misclassified(:,2),400,'x','k','Linewidth',2);
scatter(support_vectors(:,1),support_vectors(:,2),300,'+','b','Linewidth',2);
scatter(inside_margin(:,1),inside_margin(:,2),50,'^','c','filled','MarkerEdgeColor','k','Linewidth',1);
ylim([-2,6]);
hold off
title(['C = ', num2str(C(i)), '; Misclassified = ', num2str(length(total_misclassified)), '; Margin Length = ', num2str(margin), ';Support Vectors = ', num2str(length(total_support_vectors))]);
xlabel('X1');
ylabel('X2');
legend('Class1','Class2','Decision boundary','Margin1','Margin2','Misclassified','Support Vectors (On Margin)','Support Vectors (Inside Margin + Misclassifed)','Location','southeast','FontSize',12);

end

%% Part 2

% Number of experiments with different number of samples 
experiment_no = 10;
% Number of time trials for each method (values averaged later to get a single value of time)
time_no = 5;

% Initialize storage vector for time and number of samples
time_store_fitcsvm = zeros(experiment_no,1);
time_store_SOFT_SVM = zeros(experiment_no,1);
time_store_fitcsvm_L1QP = zeros(experiment_no,1);
store_sample_no = zeros(experiment_no,1);
trial_store_fitcsvm = zeros(time_no,1); % For repeating the same experiment multiple times and getting an average time
trial_store_SOFT_SVM = zeros(time_no,1); % For repeating the same experiment multiple times and getting an average time
trial_store_fitcsvm_L1QP = zeros(time_no,1); % For repeating the same experiment multiple times and getting an average time

% Initialize cell arrays for storing class 1 and class 2 samples that is generated in the below for loop
CLASS1 = cell(experiment_no,1);
CLASS2 = cell(experiment_no,1);

% Save the initially generated data from Part 1).
store1 = class1;
store2 = class2;
CLASS1{1} = store1;
CLASS2{1} = store2;

% Generate samples with different sample sizes and store in cell array
for i=2:experiment_no
    % Generate additional data
    class1=mvnrnd([1 3],[1 0; 0 1],60);
    class2=mvnrnd([4 1],[2 0; 0 2],40);
    
    % Appends the newly generated class1 and class2 data into store1 and store2 (Done to keep adding new data points to the previous datapoints rather than generating a completely new dataset)
    store1 = [store1;class1];
    store2 = [store2;class2];
    
    % Save each generated data into cell arrays for class1 and class2
    CLASS1{i} = store1;
    CLASS2{i} = store2;
end


% Experiment for 10 different sample sizes from 100-1000 and for different C = 0.1 and C = 100
for j = 1:length(C)
    
    BOXCONSTRAINT = C(j); % Set C value
    
    % Calculate computation time and save results
    for i=1:experiment_no
        
        % Use the generated samples with different sample sizes
        class1=CLASS1{i};
        class2=CLASS2{i};
        
        % Time SOFT_SVM Function (Our implementation)
        for k = 1:time_no
        tic;
        [w, wo,lambda, margin,total_support_vectors,N,class_features,labels] = SOFT_SVM(class1,class2,BOXCONSTRAINT);
        time_SOFT_SVM = toc;
        trial_store_SOFT_SVM(k,1) = time_SOFT_SVM;
        end
        time_SOFT_SVM = mean(trial_store_SOFT_SVM);
        % Store SOFT_SVM Function times
        time_store_SOFT_SVM(i,1) = time_SOFT_SVM;
        
        % Store sample size
        store_sample_no(i,1) = N;
        
        % Time fitcsvm(SMO) Function (Matlab implementation)
        for k = 1:time_no
        tic;
        SVM_Model = fitcsvm(class_features,labels,'BoxConstraint',BOXCONSTRAINT,'Solver','SMO');
        time_fitcsvm = toc;
        trial_store_fitcsvm(k,1) = time_fitcsvm;
        end
        time_fitcsvm = mean(trial_store_fitcsvm);
        % Store fitcsvm Function times
        time_store_fitcsvm(i,1) = time_fitcsvm;
        
        % Time fitcsvm(L1-quadprog) Function (Matlab implementation)
        for k = 1:time_no
        tic;
        SVM_Model = fitcsvm(class_features,labels,'BoxConstraint',BOXCONSTRAINT,'Solver','L1QP'); % Uses quadprog (Optimization Toolbox) to implement L1 soft-margin minimization by quadratic programming. 
        time_fitcsvm_L1QP = toc;
        trial_store_fitcsvm_L1QP(k,1) = time_fitcsvm_L1QP;
        end
        time_fitcsvm_L1QP = mean(trial_store_fitcsvm_L1QP);
        % Store fitcsvm Function times
        time_store_fitcsvm_L1QP(i,1) = time_fitcsvm_L1QP;
        
    end
    
    % Plot Time vs Samples for SOFT_SVM and fitcsvm(SMO)
    figure(2)
    subplot(1,2,j)
    plot(store_sample_no,time_store_SOFT_SVM,'r','LineWidth',3.0); % SOFT_SVM (Our implementation)
    hold on
    plot(store_sample_no, time_store_fitcsvm,'g','LineWidth',3.0); % fitcsvm (Matlab SMO implementation)
    plot(store_sample_no, time_store_fitcsvm_L1QP,'b','LineWidth',3.0); % fitcsvm (Matlab L1-quadprog implementation)
    hold off
    xlabel('Sample Size');
    ylabel('Time');
    title(['Time vs Sample size ; C = ',num2str(BOXCONSTRAINT)]);
    legend('SOFTSVM','fitcsvm (SMO)','fitcsvm (L1-quadprog)');
end

toc(Begin); % Calculate the elapsed time to run the whole script from beginning to end.

%% SOFT_SVM FUNCTION - SELF IMPLEMENTATION OF SOFT MARGIN SVM (Used for Part 2 of the Project)
function [w, wo,lambda, margin,total_support_vectors,N,class_features,labels] = SOFT_SVM(class1,class2,C)

% Combine class features and generate the labels for class 1 and class 2
class_features = [class1;class2];
labels = [ones(length(class1),1);-1.*ones(length(class2),1)];
N = size(class_features,1); % Size of samples

% Prepare data for quadprog by finding H,f (x = lambda)
% Note: quadprog finds the minimum for a quadratic cost function but, we need to maximize the given cost function for the dual form which causes f and H terms to be multiplied by -1.
H = (class_features*class_features').*(labels*labels');
f = -1.*ones(N,1);

% Specify the lower and upper bound
lb = zeros(N,1);
ub = repmat(C,N,1);

% Note: There are no inequality constraints for the dual form besides the lower and upper bounds so, A = [] and b = [] for (A.x <= b) in quadprog
A = [];
b =[];

% Find Aeq and beq for the constraint Aeq*x = beq (y'*lambda = 0) where lambda = x
Aeq = labels';
beq = 0;

% Set Quadprog for finding lambda
lambda = quadprog(H,f, A, b, Aeq, beq, lb,ub);
lambda(lambda < 1E-5) = 0; % set lambda = 0 for lambda < 1*10^-5
lambda = round(lambda, 4); % round the value to the nearest 4th significant digit for stability of the solution

% Find w (equation from page 126 (section 3) of the Pattern Recogniticon book)
w = (class_features'*(labels.*lambda)); 

% Find the indices of support vectors
[Sup_vec, ~] = find(lambda > 0 & lambda < C); 

% Find wo using equation 3.100 in the book (KKT conditions), and only (strictly) using the features and labels associated with support vectors (0<lambda<C) so that slack variables = eps(i) = 0 and can be removed from the equation.
% Note: Using lambda = C leads to mu(i) = 0 from the constraint C - m(i) - lambda(i) = 0 and from the constraint mu(i)*eps(i) = 0 if mu(i) = 0 then eps(i) >= 0 (not necessarily equals zero) where eps(i) are the slack variables
% Note: The point of using this trick above is because wo is expected to be the same value for all the examples in d(x) = wx(i) + wo but, we are left with 2 unknowns in equation 3.100 after finding w (wo and eps(i))
wo = mean((1./labels(Sup_vec)) - class_features(Sup_vec,:)*w);

% Calculate margin length
margin = 2/norm(w);

% Find support vectors
support_vectors = class_features(Sup_vec,:);

% Find support vectors inside margin
[inside_margin_index, ~] = find(lambda == C);
inside_margin = class_features(inside_margin_index,:);

% Total number of support vectors (lambda > 0)
total_support_vectors = [support_vectors;inside_margin];

end

