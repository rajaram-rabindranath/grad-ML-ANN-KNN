%% ========================================================================
%  Filename    : preprocess.m
%  Description : function preprocess
%  Project     : ANN vs KNN MACHINE LEARNING CSE 574 -- PROJECT 1
%  Authors     : ANGAD GADRE, HARISH MANGALAPALLI & RAJARAM RABINDRANATH
%  ========================================================================

function [train_data, train_label, validation_data,validation_label, ...
    test_data, test_label] = preprocess()


clearvars;


%% ========================================================================
%  ==========================  Load datasets  =============================
%  ========================================================================
load('mnist_all.mat'); %%%% FIXME git the path of relative path
count = 9;

%% ========================================================================
%  ============  Creating Train & Validation datasets  ====================
%  ========================================================================


%% pre-processing -- need to lose dimension
% input dataset has N X 784
% need to lose some features that do not contribute to variations
% then post losing non-important features -- have to split the training
% dataset into training and validation sets

for index=0:count
   expression = ['sumtrain',num2str(index),...
       '= sum(train',num2str(index),');']; 
   eval(expression)
end

%% append the rows begotten by roll up of each training dataset
% need to find which columns do not contri to variation across all datasets
% need to only loose those columns
sumtrain10 = vertcat(sumtrain0, sumtrain1, sumtrain2, sumtrain3,...
    sumtrain4, sumtrain5, sumtrain6, sumtrain7, sumtrain8, sumtrain9);
total = sum(sumtrain10);

%% give me the indices of the columns that have value > 0 (basically non-zero cols)
% keep only valid columns in both the training and test sets
validCols = find(total);
for index=0:count
expression = ['test',int2str(index),'_validCols','= ','test',int2str(index),'(:,validCols)'];
evalc(expression);
expression = ['train',int2str(index),'_validCols','= ','train',int2str(index),'(:,validCols)'];
evalc(expression);
end;

%% creating the training datasets and the validation datasets
% we need to split the dataset into training and validation sets
% 1/6th of each dataset shall be validation set
% 5/6th of each dataset shall be the training set
fraction_training = 5/6;
for index=0:count
    % get the num of rows of the dataset
    datasetName= ['train',num2str(index),'_validCols'];
    expression = ['nrows_dataset','=','size(',datasetName,',1);']; % return the size in the first dimension 
    evalc(expression); 
    
    % compute row counts for the training dataset and set about doing 
    % randomization for selections of records/images
    nrows_trainingSet = round(nrows_dataset*fraction_training); % count of rows in the training set
    randRows = randperm(nrows_dataset);
    
    % subset the training data into 
    % train set 
    expression =['subtrain',num2str(index),'=',datasetName,'(randRows(1:nrows_trainingSet),:);'];
    evalc(expression); 
    
    %validation set
    expression=['validate',num2str(index),'=',datasetName,'(randRows(nrows_trainingSet:end),:);'];
    evalc(expression);
end



%% appending the individual training datasets to form the consolidated training set
% and doing likewise for validation dataset
train_data = vertcat(subtrain0,subtrain1,subtrain2,subtrain3,...
    subtrain4,subtrain5,subtrain6,subtrain7,subtrain8,subtrain9);

validation_data =  vertcat(validate0,validate1,validate2,validate3,...
    validate4,validate5,validate6,validate7,validate8,validate9);

%% creating labels for the training data
for i=0:count
    eval(['label' num2str(i) '=repmat(' num2str(i) ', size(subtrain'...
        num2str(i) ', 1), 1);']);
end
label = '[';
for i=0:count-1
   label = strcat(label, 'label',  num2str(i), ';');
end
label = strcat(label, 'label',  num2str(9), '];');
train_label = eval([label]);

%% creating labels for the validation datasets
for i=0:count
    eval(['label' num2str(i) '=repmat(' num2str(i) ', size(validate' num2str(i) ', 1), 1);']);
end
label = '[';
for i=0:count-1
   label = strcat(label, 'label',  num2str(i), ';');
end
label = strcat(label, 'label',  num2str(9), '];');
validation_label = eval([label]);

%% convert dataset elements to double and normalize the same
train_data = double(train_data);
train_data = train_data/255;

%% convert dataset elements to double and normalize the same
validation_data = double(validation_data);
validation_data = validation_data/255;

%% ========================================================================
%  ========================== Creating test datasets ======================
%  ========================================================================

test_data = vertcat(test0_validCols,test1_validCols,test2_validCols, ...
    test3_validCols,test4_validCols,test5_validCols,test6_validCols, ...
    test7_validCols,test8_validCols,test9_validCols);

%% ====================== creating test labels ============================
for i=0:count
    eval(['label' num2str(i) '=repmat(' num2str(i) ', size(test' num2str(i) ', 1), 1);']);
end


label = '[';
for i=0:count-1
   label = strcat(label, 'label',  num2str(i), ';');
end
label = strcat(label, 'label',  num2str(9), '];');
test_label = eval([label]);

%% converting test data elements into double and normalizing
test_data = double(test_data);
test_data = test_data/255;


%% ========================================================================
%  ==================== Deleting unnecessary dataset ======================
%  ========================================================================

for index=0:count
    
    expression = ['clear ','test',num2str(index)];
    eval(expression)
    expression = ['clear ','test',num2str(index),'_validCols'];
    eval(expression)

    expression = ['clear ','train',num2str(index)];
    eval(expression);
    expression = ['clear ','train',num2str(index),'_validCols'];
    eval(expression)
    
    expression = ['clear ','sumtrain',num2str(index)];
    eval(expression)
    
    expression = ['clear ','label',num2str(index)];
    eval(expression)
    
    expression = ['clear ','subtrain',num2str(index)];
    eval(expression);
    expression = ['clear ','validate',num2str(index)];
    eval(expression);
end


clear validCols;
clear total;
clear sumtrain10;
clear nrows_dataset
clear nrows_trainingSet
clear expression
clear count
clear dataPath
clear datasetName
clear fraction_training
clear i
clear index
clear label

%% ========================================================================
%  ====================== FEATURE EXTRACTION, STD =========================
% =========================================================================
% test if std gives better feature selection - are there features that 
% show no variation across examples and there fore are redundant
% non contributing features i.e. features that have a zero in all the
% images
size(train_data)
std_features = std(train_data,0,1);
size(std_features)
variantCols = find(std_features);
size(variantCols)

train_data = train_data(:,variantCols);
validation_data =  validation_data(:,variantCols);
test_data = test_data(:,variantCols);

%% ========================================================================
%  ====================== ADDING BIAS COL =================================
% =========================================================================
bias_col = ones(size(train_data,1),1);
train_data = [train_data bias_col];

bias_col = ones(size(validation_data,1),1);
validation_data = [validation_data bias_col];

bias_col = ones(size(test_data),1);
test_data = [test_data bias_col];

end
% ============================ END OF FILE ================================
