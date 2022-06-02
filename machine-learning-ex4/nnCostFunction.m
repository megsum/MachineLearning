function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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

% Need to make y into a matrix of 0s and 1s depending on the value of the sample
% For example, if x(i) is 5, then the corresponding y(i) should be a 10-dimensional
% vector with y5 = 1, and the other elements equal to 0
ynew = zeros(m, num_labels);
for i = 1:m
    % y(0) doesn't exist, so we move 0 to the 10th index
    if y(i) == 0
        ynew(i, 10) = 1;
    else
        ynew(i, y(i)) = 1;
end

% Compute the cost with the new y matrix
y_one = -ynew .* log(h);
y_two = (1-ynew) .* log(1-h);

% Must sum for each sample and each label
J = (1/m) * sum(sum(y_one - y_two));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% Set initial values for Deltas so they can be accumulated
Delta1 = zeros(size(Theta1));;
Delta2 = zeros(size(Theta2));;

% You should implement steps 1 to 4 in a loop that processes one example at a time. Concretely, you should implement a for-loop for t = 1:m and
% place steps 1-4 below inside the for-loop, with the tth iteration performing the calculation on the tth training example (x(t), y(t)). Step 5 will divide the
% accumulated gradients by m to obtain the gradients for the neural network cost function.
for i = 1:m

% STEP 1. Perform a feedforward pass
% Already computed the values for step 1. Reusing them here

% STEP 2. For each output unit k in layer 3 (the output layer), set d_3 
% where yk ∈ {0, 1} indicates whether the current training example belongs to class k (yk = 1), or if it belongs to a different class (yk = 0).
% You may find logical arrays helpful for this task (explained in the previous programming exercise).
    d3 = a3(i,:) - ynew(i,:);

% STEP 3. For the hidden layer l = 2, set d_2
    d2 = d3 * Theta2;
    d2 = d2(2:end);
    d2 = d2 .* sigmoidGradient(z2(i,:));

% STEP 4. Accumulate the gradient from this example using the following formula. Note that you should skip or remove δ(2)0
% In Octave/MATLAB,removing δ(2)0 corresponds to d2 = d2(2:end)
% Not too sure why the d variables need to be transposed. BUT IT WORKED :D
    Delta1  = Delta1  + d2' * a1(i,:);
    Delta2 = Delta2 + d3' * a2(i,:);
end

% STEP 5. Obtain the (unregularized) gradient for the neural network cost function by dividing the accumulated gradients by 1 / m
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Calculate the regularization cost. Don't do regularization on the bias layer
Theta1_reg = sum(sum(Theta1(1:end,2:input_layer_size+1).^2));
Theta2_reg = sum(sum(Theta2(1:end,2:hidden_layer_size+1).^2));
J_reg = lambda / (2 * m) *  (Theta1_reg + Theta2_reg);

% Add the regularization cost to our calculated cost
J = J + J_reg;

% Add the regularization gradient to our calculated gradient
Theta1_grad(:,2:end) = Theta1_grad + Theta1_reg + (lambda/m) * Theta1;
Theta2_grad(:,2:end) = Theta2_grad + Theta2_reg + (lambda/m) * Theta2;;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
