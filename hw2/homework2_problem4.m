% gradient descent function
function [x, funcVal] = homework2_problem4(func, x0, Ytrain, Xtrain)
% func: cost function, can be evaluate at given x, must return both value
% and gradient evaluated at given x, x is of size n x k
% x0: starting point, matrix, of size n x k, means n points in k dimension
    learningRate = 0.0001;
    iterMax = 1000;
    threshold = 1e-4;
    gap = 1000;
    
    x = x0;
    iterTimes = 0;
    while gap>threshold && iterTimes < iterMax     
        [funcVal, grad] = feval(func, x, Ytrain, Xtrain);
        xNew = x - learningRate*grad;
        gap = norm(Xtrain*xNew - Ytrain, 2);
        x = xNew;
        iterTimes = iterTimes + 1;
    end
end
