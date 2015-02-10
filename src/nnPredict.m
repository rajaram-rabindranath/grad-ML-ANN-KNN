function label = nnPredict(w1, w2, train_data)
% nnPredict predicts the label of data given the parameter w1, w2 of Neural
% Network.

% Input:
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit i in input 
%     layer to unit j in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit i in hidden 
%     layer to unit j in output layer.
% data: matrix of data. Each row of this matrix represents the feature 
%       vector of a particular image + 1 bias feature
       
% Output: 
% label: a column vector of predicted labels
    
    z = [w1 * train_data'; ones(1, size(train_data, 1))];
    
    
    
    y = w2 * sigmoid(z);
    
    h = sigmoid(y);
    
    [~,label] = max(h);
    label = label - 1;
    label = label';
end
