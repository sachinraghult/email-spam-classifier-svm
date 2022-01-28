function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

cVec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];      %8x1
sigmaVec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];  %8x1

error = zeros(size(cVec), size(sigmaVec));        %8x8


for i = 1 : size(cVec),
  for j = 1 : size(sigmaVec),
    cCur = cVec(i);
    sigmaCur = sigmaVec(j);
    
    model= svmTrain(X, y, cCur, @(x1, x2) gaussianKernel(x1, x2, sigmaCur));
    predictions = svmPredict(model, Xval);
    
    error(i, j) = mean(double(predictions ~= yval));
  endfor
endfor

%errorMin is 1x8 where it contains min of each column values and cMin has indexes of those minimum values
%best is the final minimum error that is found and sigmaMin is the index of that minimal error

[errorMin, cMin] = min(error);
[best, sigmaMin] = min(errorMin);

%Hence C is value in cVec at index cMin(row_index) where sigmaMin(col_index) is found
%And sigma is the value in sigmaVec at sigmaMin

C = cVec(cMin(sigmaMin));
sigma = sigmaVec(sigmaMin);


% =========================================================================

end
