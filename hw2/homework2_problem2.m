% problem 2.1
% loss function of L1 and L2 regularization:
% cost function J = (Y - Xb)'(Y-Xb)

% problem 2.3
climateChangeData = readtable('climate_change_1.csv','PreserveVariableName',1);
trainData = climateChangeData{climateChangeData.Year<=2006,:};
testData = climateChangeData{climateChangeData.Year>2006,:};

%create X and add constant column
Xtrain = trainData(:,3:10);
Xtrain = [ones(size(Xtrain,1),1),Xtrain];

Xtest = testData(:,3:10);
Xtest = [ones(size(Xtest,1),1),Xtest];

%create Y
Ytrain = trainData(:,11);
Ytest = testData(:,11);

beta_OLS = closed_form_1(Ytrain, Xtrain);
beta_L2 = closed_form_2(Ytrain, Xtrain, 10);

% display
disp("norm of betas:");
disp(norm(beta_OLS,2));
disp(norm(beta_L2,2));

% comment: in L2 regularization, the norm of beta(coefficient vector) is
% restricted, thus the optimal solution won't be influenced too much by
% irregular data.

% problem 2.4
% with k-fold cross validation, cut training data into training subset and
% validation subset. According to the problem, the first job is just to
% check R^2 of different lambdas on both whole training and testing data.

% first job
lambdaArray = [0.001,0.01,0.1,1,10];
for count = 1:length(lambdaArray)
    lambdaStruct = workFlow(Xtrain, Xtest, Ytrain, Ytest, lambdaArray(count));
    fN = "lambda" + num2str(count);
    R2Struct.(fN) = lambdaStruct;
end

% second job, use k-fold cv get a best model among given lambdas
% simple strategy, for each lambda, sample k times, each time, 
% cut m numbers in training data as a validation set, the rest as training 
% subset. run different lambdas, choose the model has minimum mean squared
% error on validation set. Deploy the model on the test data.

% get best lambda
k = 5;
m = 30;
bestLambda = kFoldCV(Xtrain, Ytrain, k, m, lambdaArray);

% deploy best lambda on the test data
cvLambdaStruct = workFlow(Xtrain, Xtest, Ytrain, Ytest, bestLambda);
cvTrainR2 = cvLambdaStruct.train;
cvTestR2 = cvLambdaStruct.test;

% problem 2.2
% solve L2 regularization
function betas = closed_form_2(Y,X,lambda)
    I = ones(size(X'*X));
    betas = (X'*X + lambda * I)\(X'*Y);
end

% solve OLS
function betas =  closed_form_1(Y,X)
    betas = (X'*X)\(X'*Y);
end

% universal function, get different lambda(a scalar), return R^2 of
% training and testing data set
function R2TrainTest = workFlow(Xtrain, Xtest, Ytrain, Ytest, lambda)
    %get betas from training
    betas = closed_form_2(Ytrain, Xtrain, lambda);
    Yhat = Xtrain*betas;
    
    %get R2 of training set
    SSR_train = sum((Yhat - mean(Ytrain)).^2);
    SST_train = sum((Ytrain - mean(Ytrain)).^2);
    R2train = SSR_train / SST_train;
    
    %get Yhat of testing data set
    Yhat_test = Xtest*betas;
    
    %get R2 of test set
    SSR_test = sum((Yhat_test - mean(Ytest)).^2);
    SST_test = sum((Ytest - mean(Ytest)).^2);
    R2test = SSR_test / SST_test;
    
    R2TrainTest.lambda = lambda;
    R2TrainTest.train = R2train;
    R2TrainTest.test = R2test;
end

% k-fold cross validation
function bestLambda = kFoldCV(Xtrain, Ytrain, k, m, lambdaArray)
    %for each lambda do:
    minMeanMSE = inf;
    bestLambda = lambdaArray(1);
    for lambda = lambdaArray
        kTimesMSE = zeros(1,k);
        % do k times evaluation
        for time = 1:k
            % cut training data
            shuffleIndex = randperm(size(Xtrain,1));
            XtrainSubset = Xtrain(shuffleIndex(m+1:end),:);
            YtrainSubset = Ytrain(shuffleIndex(m+1:end));
            Xvalidation = Xtrain(shuffleIndex(1:m),:);
            Yvalidation = Ytrain(shuffleIndex(1:m));
            
            % run regression on train subset, get betas
            betas = closed_form_2(YtrainSubset, XtrainSubset, lambda);
            % test MSE on validation data set
            SSE = sum((Yvalidation -  Xvalidation * betas).^2);
            kTimesMSE(time) = SSE / m;
        end
        % compare kTimes mean MSE
        meanMSE = mean(kTimesMSE);
        if meanMSE < minMeanMSE
            minMeanMSE = meanMSE;
            bestLambda = lambda;
        end
    end
end