% use LASSO to select features 
% first set a loss measure, 1SE here(the variables in the sparsest model 
% within one standard error of the minimum MSE.), to select potential models
% (with different lambda)
% use k-Fold cross validation, choose best lambda through minimize MSE and
% deplot the model on the test data set.
climateChangeData = readtable('climate_change_1.csv','PreserveVariableName',1);
trainData = climateChangeData{climateChangeData.Year<=2006,:};
testData = climateChangeData{climateChangeData.Year>2006,:};

%create X and add constant column
Xtrain = trainData(:,3:10);
% Xtrain = [ones(size(Xtrain,1),1),Xtrain];

Xtest = testData(:,3:10);
% Xtest = [ones(size(Xtest,1),1),Xtest];

%create Y
Ytrain = trainData(:,11);
Ytest = testData(:,11);

[B,FitInfo] = lasso(Xtrain,Ytrain,'CV',10);

lassoPlot(B,FitInfo,'PlotType','CV');
legend('show'); % Show legend

% fit model with new parameters
bestBetas = B(:,FitInfo.Index1SE);

% get R2 of test data
Yhat_test = Xtest*bestBetas + FitInfo.Intercept(FitInfo.Index1SE);
SST_test = sum((Ytest - mean(Ytest)).^2);
SSR_test = sum((Yhat_test - mean(Ytest)).^2);
R2test_LASSO = SSR_test / SST_test;
R2test_Ridge = 0.5817;
