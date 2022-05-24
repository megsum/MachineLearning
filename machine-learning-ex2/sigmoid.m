function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% The sigmoid function is defined as g(z) = 1 / (1 + e^-z)
% First, populate a matrix the size of z with e
e_var = ones(size(z));
e_var = e * e_var;

g = 1 ./ (1 + e_var.**-z)
% =============================================================

end
