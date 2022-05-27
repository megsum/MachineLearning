function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X * theta);

y_zero = (-y .* log(h));
y_one = ((1 - y) .* log(1 - h));

% Calculate regularized cost
J = (1 / m) * sum(y_zero - y_one) + (lambda / (2 * m)) * sum(theta(2:end).^2);

for j = 1:length(theta)
    % Should not regularize when theta = 0
    if j == 1
        % Calculate cost for theta = 0
        %J = (1 / m) * sum(y_zero - y_one);
        % Calculate gradient for theta = 0
        grad(1) = (1 / m) *  sum((h - y) .* X(:,1));

    else
        % Calculate regularized gradient
        grad(j) = (1 / m) *  sum((h - y) .* X(:,j)) + (lambda / m) * theta(j);
end

% =============================================================

end
