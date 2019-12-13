% problem 1.1
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

%get closed form solution
betas = closed_form_1(Ytrain,Xtrain);

%problem 1.2
% mathemactical formula of linear model
% $Y = X\beta + \epsilon$, where $\epsilion \sim^{i.i.d.} N(0,\sigma^2)$
% suppose no outliers exist and Y and X has linear relation

whichstats = {'rsquare'};
% model R^2 of training model
trainRSquared = regstats(Ytrain, Xtrain, 'linear', whichstats);
trainRSquared = trainRSquared.rsquare;

% model R^2 of test model
YtestHat = Xtest*betas;
SSR = sum((YtestHat - mean(Ytest)).^2);
SST = sum((Ytest - mean(Ytest)).^2);
testRSquared = SSR / SST;

% problem 1.3
% Hypothesis testing H0:beta = 0
whichstats = {'tstat'};

model = regstats(Ytrain, Xtrain, 'linear', whichstats);
modelPValue = model.tstat.pval;
indxSignificant = find(modelPValue<0.05);

% problem 1.4
% to use closed form solution, we must have (X'*X) is invertible
% apply same method to the climate_change_2.csv

% read another table
climateChangeData = readtable('climate_change_2.csv','PreserveVariableName',1);

% use same rule to generate train data
trainData = climateChangeData{climateChangeData.Year<=2006,:};
testData = climateChangeData{climateChangeData.Year>2006,:};

%create X and add constant column
Xtrain = trainData(:,3:11);
Xtrain = [ones(size(Xtrain,1),1),Xtrain];

Xtest = testData(:,3:11);
Xtest = [ones(size(Xtest,1),1),Xtest];

%create Y
Ytrain = trainData(:,12);
Ytest = testData(:,12);

%get closed form solution
betas = closed_form_1(Ytrain,Xtrain);

%why solution bad:
%because rank(X'X) = 9, while size(X'X) = 10 x 10

function betas =  closed_form_1(Y,X)
    betas = (X'*X)\(X'*Y);
end
