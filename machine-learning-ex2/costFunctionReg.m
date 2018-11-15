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

h = sigmoid(X*theta);                         % Sigmoid for cost function.

predictions = [-y'*log(h)-(1-y')*log(1-h)];   % Difference between calculated
                                              % values and original values.


% Cost calculation when features are regularized.
% The first parameter of theta matrix is not accounted for.
J = 1/m * sum(predictions) + lambda/(2*m) * sum(theta([2:size(X,2)],1).^2);


% Gradient for cost function when features are regularized.
grad(1,:) = [(1/m) * X'(1,:)*(h - y)];
grad([2:size(X,2)],:) = [(1/m) * X'([2:size(X,2)],:)*(h - y)]...
+ (lambda/m) * theta([2:size(X,2)],:) ;

% =============================================================

end
