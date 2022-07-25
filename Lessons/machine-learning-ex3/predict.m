function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Input layer is a1 (X)
a1 = X;

% Add ones to signify the bias unit 
a1 = [ones(m, 1) a1];

% z2 is the first hidden layer. We can use our initial theta for it
z2 = a1 * Theta1';
a2 = sigmoid(z2);
% Add ones to signify the bias unit
a2 = [ones(m, 1) a2];

% z3 is the second hidden layer. We use the thetas computed by the project
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Our hypothesis is equal to layer 3 (output layer)
h = a3;

% Get the number that is most likely based off the hypothesis
[max_num, index] = max(h, [], 2);

% Index can be used to determine which number is likely (from 1 to 10 with 0 being equal to 10)
p = index;

% =========================================================================


end
