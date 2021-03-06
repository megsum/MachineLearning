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

C_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

prev_error = 1000;
% Loop through all the potential values of C
for i = 1: length(C_vals)
    % Loop through all the potential values of sigma
    for j = 1: length(sigma_vals)
        % Train the given model with the potential values on a gaussian curve
        model= svmTrain(X, y, C_vals(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vals(j)));
        predictions = svmPredict(model, Xval);
        cur_error = mean(double(predictions ~= yval));
        % check if the previous error is greater than the current predicted error
        if prev_error > cur_error
            % If so, our current error is better and we should update the values of sigma and C
            prev_error = cur_error;
            C = C_vals(i);
            sigma = sigma_vals(j);
    end
end
% =========================================================================

end
