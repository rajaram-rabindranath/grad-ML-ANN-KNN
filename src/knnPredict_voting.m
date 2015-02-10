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

    %% since we have the bias col already added to both the train and test data
    numFeatures = size(train_data,2)-1;

    test_data_ = test_data(:,1:numFeatures);
    train_data_ = train_data(:,1:numFeatures);

    %% Initialize label matrix
    label = zeros(numTestInst,1);

    %% KNN
    [D, I] = pdist2(train_data_, test_data_,'cosine','Smallest',k);
    test_label_ = train_label(sub2ind([numTrainInst k], I, ones(k, numTestInst)), 1);
    predicted_labels = reshape(test_label_, k, numTestInst);
    U = unique(predicted_labels);
    H=histc(predicted_labels, U, 1);
    maxH=max(H,1);
    for i = 1:numTestInst
        idx = find(H(:,i) == maxH(1,i));
        idx = idx - 1;
        if(size(idx, 1)>1)
            d = zeros(size(idx, 1));
            for j=1:size(idx)
                localidx = find(predicted_labels(:,i) == idx(j));
                d(j) = sum(D(localidx,i));
            end
            [~,ind] = min(d);
            label(i) = idx(ind);
        else
            label(i) = idx;
        end
    end
    %label = mode(predicted_labels, 1)';
end

