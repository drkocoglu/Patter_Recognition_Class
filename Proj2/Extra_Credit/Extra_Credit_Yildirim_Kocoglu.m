% Extra Credit (Pattern Recognition)
% Name: Yildirim Kocoglu

clear;
clc;
close all;

%% Import data

% Generate the required data for class 1 and class 2
rng(100); 
class1=mvnrnd([1 3],[1 0; 0 1],60); 
class2=mvnrnd([4 1],[2 0; 0 2],40);

% Given C values
C = [10,100];

% Given sigma value
sigma1 = 1.75;

%% Extra Credit - (Gaussian Kernel-SOFT Margin SVM)

% Combine class features and generate the labels for class 1 and class 2
class_features = [class1;class2];
labels = [ones(length(class1),1);-1.*ones(length(class2),1)];
N = size(class_features,1);

% Solve lambda, w*x, and wo using quadprog for C=10 and C=100.
for k = 1:2 % for C=10 and C=100

% Initialize RBF Kernel
Kernel = zeros(N,N);

% Gaussian (RBF) Kernel
for i = 1:N
    Kernel(:,i) = exp((-vecnorm(class_features-class_features(i,:),2,2).^2)./(2*(sigma1^2)));
end

% Prepare data for quadprog by finding H,f (x = lambda)
% Note: quadprog finds the minimum for a quadratic cost function but, we need to maximize the given cost function for the dual form which causes f and H terms to be multiplied by -1.
H = Kernel.*(labels*labels');
f = -1.*ones(N,1);

% Specify the lower and upper bound
lb = zeros(N,1);
ub = repmat(C(k),N,1); % C = 10 or C = 100

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

% Find the indices of support vectors
[Sup_vec, values] = find(lambda > 0 & lambda < C(k)); 

wx = (Kernel*(labels.*lambda));

% Find wo using equation 3.100 in the book (KKT conditions), and only (strictly) using the features and labels associated with support vectors (0<lambda<C) so that slack variables = eps(i) = 0 and can be removed from the equation.
% Note: Using lambda = C leads to mu(i) = 0 from the constraint C - m(i) - lambda(i) = 0 and from the constraint mu(i)*eps(i) = 0 if mu(i) = 0 then eps(i) >= 0 (not necessarily equals zero) where eps(i) are the slack variables
% Note: The point of using this trick above is because wo is expected to be the same value for all the examples in d(x) = wx(i) + wo but, we are left with 2 unknowns in equation 3.100 after finding w (wo and eps(i))
wo = mean((1./labels(Sup_vec)) - wx(Sup_vec));

% Find support vectors on margin
Sup_vectors_on_margin = class_features(Sup_vec,:);
% Find support vectors inside margin + misclassified
[inside_margin_index, inside_margin_values] = find(lambda == C(k));
Inside_margin_misclassified = class_features(inside_margin_index,:);

% Total number of support vectors (lambda > 0)
total_support_vectors = [Sup_vectors_on_margin;Inside_margin_misclassified];

% Find misclassified features
dx = (Kernel*(lambda.*labels))+wo;
[misclassified_index,missclass_value] = find(dx(1:length(class1)) < 0); 
misclassified = class_features(misclassified_index,:);
[misclassified_index_2,missclass_value_2] = find(dx(length(class1)+1:length(class_features)) > 0);
misclassified_index_2 = misclassified_index_2 + length(class1);
misclassified_2 = class_features(misclassified_index_2,:);
total_misclassified = [misclassified;misclassified_2];

% Draw the non-linear decision boundary

% Create meshgrid (test data) for contour drawing (plotting) decision boundaries
x1 = [floor(min(class_features(:,1))-2), floor((max(class_features(:,1))+2))]; % (+-)2 added to draw the full decision boundary

x2 = [floor(min(class_features(:,2))-2), floor((max(class_features(:,2)))+3)]; % -2,+3 added to draw the full decision boundary

[X1,X2] = meshgrid(linspace(x1(1), x1(2), 100), linspace(x2(1), x2(2), 100));
XX1=reshape(X1,[],1);
XX2=reshape(X2,[],1);
Mesh_Grid=[XX1,XX2];

% Initialize testing Kernel (used after solution is derived from the training data (given 100 samples)
Kernel_test = zeros(N*N,N);

% Calculate the test Kernel (test data)
for i = 1:N
    Kernel_test(:,i)= exp(-(vecnorm(Mesh_Grid-class_features(i,:),2,2).^2)./(2*sigma1^2) );
end

% Non Linear Decison Functions
gx = (Kernel_test*(lambda.*labels))+wo; % Decision Boundary
gx1 = (Kernel_test*(lambda.*labels))+wo + 1; % Margin 1
gx2 = (Kernel_test*(lambda.*labels))+wo - 1; % Margin 2
% Reshape decision functions (gx,gx1,gx2) for plotting contours (decision boundaries)
Z = reshape(gx,size(X1)); 
Z1 = reshape(gx1,size(X1));
Z2 = reshape(gx2,size(X1));

% Plot Decision boundary and Margins
figure (1)
subplot(1,2,k)
scatter(class_features(1:60,1), class_features(1:60,2), 200,'o', 'r', 'filled','MarkerEdgeColor','k','LineWidth',1);
hold on
scatter(class_features(61:100,1), class_features(61:100,2), 200,'s', 'g', 'filled','MarkerEdgeColor','k','LineWidth',1);
scatter(Sup_vectors_on_margin(:,1),Sup_vectors_on_margin(:,2),300,'+','b','Linewidth',2);
scatter(Inside_margin_misclassified(:,1),Inside_margin_misclassified(:,2),100,'^','y','filled','MarkerEdgeColor','k','Linewidth',1);
scatter(total_misclassified(:,1),total_misclassified(:,2),400,'x','k','Linewidth',2);
contour(X1,X2,Z,[0 0], 'k', 'LineWidth',3.0);
contour(X1,X2,Z1,[0 0], '-.b');
contour(X1,X2,Z2,[0 0], '-.r');
% limits chosen to draw full decision boundary
ylim([-4,8]);
xlim([-4,8]);
hold off
title(['C = ', num2str(C(k)), '; Misclassified = ', num2str(length(total_misclassified)), ';Support Vectors = ', num2str(length(total_support_vectors))]);
xlabel('X1');
ylabel('X2');
legend('Class1','Class2','Support Vectors (On Margin)','Support Vectors (Inside Margin + Misclassifed)','Misclassified','Decision boundary','Margin1','Margin2','Location','northeast','FontSize',10);
end