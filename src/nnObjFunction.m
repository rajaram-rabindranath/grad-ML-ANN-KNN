function [obj_val, obj_grad] = nnObjFunction(params, n_input, n_hidden, ...
                                    n_class, train_data,...
                                    train_label, lambda)
% nnObjFunction computes the value of objective function (negative log 
%   likelihood error function with regularization) given the parameters 
%   of Neural Networks, thetraining data, their corresponding training 
%   labels and lambda - regularization hyper-parameter.

% Input:
% params: vector of weights of 2 matrices w1 (weights of connections from
%     input layer to hidden layer) and w2 (weights of connections from
%     hidden layer to output layer) where all of the weights are contained
%     in a single vector.
% n_input: number of node in input layer (not include the bias node)
% n_hidden: number of node in hidden layer (not include the bias node)
% n_class: number of node in output layer (number of classes in
%     classification problem
% training_data: matrix of training data. Each row of this matrix
%     represents the feature vector of a particular image
% training_label: the vector of truth label of training images. Each entry
%     in the vector represents the truth label of its corresponding image.
% lambda: regularization hyper-parameter. This value is used for fixing the
%     overfitting problem.
       
% Output: 
% obj_val: a scalar value representing value of error function
% obj_grad: a SINGLE vector of gradient value of error function
% NOTE: how to compute obj_grad
% Use backpropagation algorithm to compute the gradient of error function
% for each weights in weight matrices.
% Suppose the gradient of w1 is 

%% create weight matrices(w1,w2) from the params vector
% reshape 'params' vector into 2 matrices of weight w1 and w2
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(hidden,inputData) 50 x featureSet_count
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(class,hidden) 10x51

% thing to note here is that : every layer shall have a bias node !!
% and this bias node is independent of bias node in the previous layer

% we have a long vector that is params and we are creating this matrix



w1 = reshape(params(1:n_hidden * (n_input + 1)),n_hidden, (n_input + 1));
w2 = reshape(params((1 + (n_hidden * (n_input + 1))):end),n_class, (n_hidden + 1));

%% number of training examples
numExamples  = size(train_data,1);
%% creating matrix of zeroes
% this matrix is out truth matrix -- holds the actual values of each and
% every example -- we shall compare the acutals with the output
% t is nothing but the matrix form of train_label -- though instead of the
% label we shall have a bit array [1,0,0,0,0,0,0,0,0] -- indicating yes for zero

actuals = zeros((n_input + 1), n_class); % create a matrix of zeros
for N = 1 : size(train_data, 1) % from 1 to num rows in train_data (examples)
    actuals(N, train_label(N) + 1) = 1; 
end

% wonder what gets stored in the last row of 't'

%% creating matrices filled with zeros
% 10x51 and 50x717 -- why are we doing this
sum_w2 = zeros(n_class, (n_hidden + 1));
sum_w1 = zeros(n_hidden, (n_input + 1));

%% Here is what we are doing:
% this matrix shall store the error values calculated using the negative
% log-likelihood function -- for each training example
% classes x TrainExamples 
nlogLikeliHood_err_n = zeros(n_class, numExamples);

%==========================================================================
% Running perceptron
% 
%==========================================================================
%  w2(class,hidden) 10x51
%  w1(hidden,inputData) 50 x featureSet_count

% train the perceptron by going thru all training examples
for x = 1 : numExamples
    % extract instance vectors -- training instance and the actual labels
    trainInstance = train_data(x,:)';
    actualsInstance =  actuals(x,:)';
    
    % -------------------- feed forward at work
        % hidden layer ops
        hiddenLayer_output = w1 * trainInstance;
        hiddenLayer_output_act = [sigmoid(hiddenLayer_output); 1];

        % output layer ops
        b = w2 * hiddenLayer_output_act;
        percetron_output = sigmoid(b);

        % lets compare predictions and actuals
        delta = percetron_output - actualsInstance; 
        % we are calculating the negaitve log-likelihood -- no summations done yet 
    
    % ------------------- back propagation at work
        nlogLikeliHood_err_n(:,x) = -1*( actualsInstance.* log(percetron_output)...
            + (1 - actualsInstance) .* log(1 - percetron_output));
    
        % back_prop for the ultimate layer -- for 10 output nodes
        % drivatives of the weights for 2nd layer (w2)-- 10x51
        gradDesc_w2_n = delta * hiddenLayer_output_act';

        % back_prop for the pen-ultimate layer
        % derivatives for the weights 1st layer (w1) -- 50 x feature_count
        sum_dk =  delta' * w2(:,1:n_hidden);
        gradDesc_w1_n = ((1 - hiddenLayer_output_act(1:n_hidden)) .* ...
            hiddenLayer_output_act(1:n_hidden)) .* sum_dk' * trainInstance';
        
    sum_w2 = sum_w2 + gradDesc_w2_n;
    sum_w1 = sum_w1 + gradDesc_w1_n;
end

%% cummulative erros and regularization done here

grad_w2 = (1/numExamples)*(sum_w2 + lambda*w2); % equation 16 the new gradiant for w2
grad_w1 = (1/numExamples)*(sum_w1 + lambda*w1); % equation 17 the new gradiant for w1

% Suppose the gradient of w1 and w2 are stored in 2 matrices grad_w1 and grad_w2
% Unroll gradients to single column vector
obj_grad = [grad_w1(:) ; grad_w2(:)];

% squashing the error matrix into a single scalar 
% do a roll up then squash the vector into a single scalar
nlogLikeliHood_err = sum(sum(nlogLikeliHood_err_n, 1), 2);

obj_val = (1/numExamples) * (nlogLikeliHood_err + (lambda/2)*(sum(sum(w1.^2,1),2) + sum(sum(w2.^2,1),2)));
end
