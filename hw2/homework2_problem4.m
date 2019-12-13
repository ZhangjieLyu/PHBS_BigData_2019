% gradient descent function
function [x, funcVal] = homework2_problem4(func, x0)
% func: cost function, can be evaluate at given x, must return both value
% and gradient evaluated at given x, x is of size n x k
% x0: starting point, matrix, of size n x k, means n points in k dimension
    learningRate = 0.1;
    iterMax = 1000;
    threshold = 1e-6;
    gap = inf;
    
    x = x0;
    while gap>threshold && iterTimes < iterMax
        [funcVal, grad] = func(x);
        xNew = x - learningRate*grad;
        gap = norm(xNew - x, 2);
        x = xNew;
    end
end