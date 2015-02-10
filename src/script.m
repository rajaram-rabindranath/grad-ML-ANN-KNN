%% ========================================================================
%  Filename    : script.m
%  Description :
%  Project     : ANN vs KNN MACHINE LEARNING CSE 574 -- PROJECT 1
%  Authors     : ANGAD GADRE, HARISH MANGALAMPALLI & RAJARAM RABINDRANATH
%  ========================================================================


%% moving to data directory
clearvars;
dataPath = 'E:\ML\base code';
cd(dataPath)
clear dataPath;

%% ========================================================================
%  ==================== PREPROCESS THE DATASET ============================
%  ========================================================================

% Feature selection: each image has 784 pixels -- don't need all of these
% pixels for classification -- need to only retain features that have a
% degree of variation across all instances of all images -- the whole deal
% create train, validation and test datasets along with labels for the same
[train_data, train_label, validation_data,validation_label, test_data,...
    test_label] = preprocess();

save('dataset.mat', 'train_data', 'train_label', 'validation_data',...
    'validation_label', 'test_data', 'test_label');
load('dataset.mat');


%% ========================================================================
%  =============== INITIALIZE PARAMETERS FOR ANN ==========================
%  ========================================================================
% set the number of nodes in input unit (not including bias unit)
% define the variables that the neural network must rely on
% count of hidden nodes = 50 nodes + 1 bias node for the next level
% count of output nodes = 10 output nodes since we have 10(0...9) 
% count of input nodes  = number of features that we shall use

n_input = size(train_data, 2)-1; 
n_hidden = 50; 
n_class = 10; 
lambda = 100;

% set the maximum number of iteration in conjugate gradient descent
options = optimset('MaxIter', 50);

%% ======================================================================== 
% ========================= KICK START ANN ================================
%% ========================================================================
iterations = 15;
% to store lambda,validation accuracy, test accuracy
ann_results = zeros(iterations,5);
step = 50;

fprintf('---------------------- STARTING ANN ----------------------\n')

for index=1:iterations

    
    ann_results(index,1) = lambda;
% initialize weights(randomly generated) into some matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

% unroll 2 weight matrices into single column vector -- why are we
% unrolling -- what has happened is
initialWeights = [initial_w1(:); initial_w2(:)];

% define the objective function
objFunction = @(params) nnObjFunction(params, n_input, n_hidden,n_class,...
    train_data, train_label, lambda);
tic    
% run neural network training with fmincg
[nn_params, cost] = fmincg(objFunction, initialWeights, options);

% reshape the nn_params from a column vector into 2 matrices w1 and w2
w1 = reshape(nn_params(1:n_hidden * (n_input + 1)), ...
                 n_hidden, (n_input + 1));

w2 = reshape(nn_params((1 + (n_hidden * (n_input + 1))):end), ...
                 n_class, (n_hidden + 1));

%   Test the computed parameters
predicted_label = nnPredict(w1, w2, train_data);
fprintf('\nTraining Set Accuracy: %f\n', ...
         mean(double(predicted_label == train_label)) * 100);

         ann_results(index,2) = mean(double(predicted_label == train_label)) * 100;
         
%   Test Neural Network with validation data
predicted_label = nnPredict(w1, w2, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', ...
         mean(double(predicted_label == validation_label)) * 100);

     ann_results(index,3) = mean(double(predicted_label == validation_label)) * 100;
     
%   Test Neural Network with test data
predicted_label = nnPredict(w1, w2, test_data);
fprintf('\nTesting Set Accuracy: %f\n', ...
         mean(double(predicted_label == test_label)) * 100);
     
    ann_results(index,4) = mean(double(predicted_label == test_label)) * 100;
 % save time taken
 ann_results(index,5)=toc     
     lambda = lambda + 50;
end

% pick the lambda that gave the highest accuracy with validation data
[~,index] = max(ann_results(:,3));
lambda = ann_results(index,1);

fprintf('=================== best lambda %d =========\n',lambda);
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

% unroll 2 weight matrices into single column vector -- why are we
% unrolling -- what has happened is
initialWeights = [initial_w1(:); initial_w2(:)];

% define the objective function
objFunction = @(params) nnObjFunction(params, n_input, n_hidden,n_class,...
    train_data, train_label, lambda);
    
% run neural network training with fmincg
[nn_params, cost] = fmincg(objFunction, initialWeights, options);

% reshape the nn_params from a column vector into 2 matrices w1 and w2
w1 = reshape(nn_params(1:n_hidden * (n_input + 1)), ...
                 n_hidden, (n_input + 1));

w2 = reshape(nn_params((1 + (n_hidden * (n_input + 1))):end), ...
                 n_class, (n_hidden + 1));

%   Test the computed parameters
predicted_label = nnPredict(w1, w2, train_data);
fprintf('\nTraining Set Accuracy: %f\n', ...
         mean(double(predicted_label == train_label)) * 100);

%   Test Neural Network with validation data
predicted_label = nnPredict(w1, w2, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', ...
         mean(double(predicted_label == validation_label)) * 100);

     
%   Test Neural Network with test data
predicted_label = nnPredict(w1, w2, test_data);
fprintf('\nTesting Set Accuracy: %f\n', ...
         mean(double(predicted_label == test_label)) * 100);


%% ========================================================================
% ==================== K-Nearest Neighbors ================================
%==========================================================================
%round(sqrt(size(train_data, 1)))
iterations = 100;
knn_results = zeros(iterations,3);


fprintf('---------------------- Starting KNN ---------------------------------\n')
for k =1:iterations
    fprintf('================== for k %d ==============\n',k)
    fprintf('Working on validation data:\n');
    tic
%   Test KNN with validation data
predicted_label = knnPredict(k, train_data, train_label, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', ...
         mean(double(predicted_label == validation_label)) * 100);
    knn_results(k,3) = toc
    knn_results(k,1) = k;
    knn_results(k,2) = mean(double(predicted_label == validation_label)) * 100;
    
end

[~,index] = max(knn_results(:,2));
k = knn_results(index,1);

fprintf('============== The best value of k %d ================\n',k)

% have the best value for K -- running it or validate N test data to get results
predicted_label = knnPredict(k, train_data, train_label, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', ...
         mean(double(predicted_label == validation_label)) * 100);
     
%   Test KNN with test data
predicted_label = knnPredict(k, train_data, train_label, test_data);
fprintf('\nTesting Set Accuracy: %f\n', ...
         mean(double(predicted_label == test_label)) * 100);


%% ========================================================================
% ========================= SAVE LEARNINGS ================================
% =========================================================================
save('params.mat', 'n_input', 'n_hidden', 'w1', 'w2', 'lambda', 'k','knn_results','ann_results');