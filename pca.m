function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

[m, n] = size(X);

U = zeros(n);
S = zeros(n);

% I first computed the covariance matrix. Then, I used the "svd" function 
% to compute the eigenvectors and eigenvalues of the covariance matrix. 


sigma = (X'*X)/m;
[U,S,V] = svd(sigma);


end
