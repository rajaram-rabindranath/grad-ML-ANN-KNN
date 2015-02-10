function label = knnPredict(k, train_data, train_label, test_data)
% knnPredict predicts the label of given data by using k-nearest neighbor
% classification algorithm

% Input:
% k: the parameter k of k-nearest neighbor algorithm
% data: matrix of data. Each row of this matrix represents the feature 
%       vector of a particular image

% Output:
% label: a column vector of predicted labels

%% instances of train and test data 
numTestInst = size(test_data,1);
numTrainInst = size(train_data,1);

% since we have the bais col already added to both the train and test data
numFeatures = size(train_data,2)-1;

test_data_ = test_data(:,1:numFeatures);
train_data_ = train_data(:,1:numFeatures);

%% KNN
[~, I] = pdist2(train_data_, test_data_,'cosine','Smallest',k);
test_label_ = train_label(sub2ind([numTrainInst k], I, ones(k, numTestInst)), 1);
predicted_labels = reshape(test_label_, k, numTestInst);
label = mode(predicted_labels, 1)';

end

