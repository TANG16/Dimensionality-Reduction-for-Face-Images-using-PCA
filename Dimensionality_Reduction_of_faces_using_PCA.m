%  Eigenfaces

clear ; close all; clc

% =============== Loading and Visualizing Face Data =============
%  Load Face dataset
load ('faces.mat')

%  Display the first 100 faces in the dataset
displayData(X(1:100, :));


% ================= PCA on Face Data ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  I displayed the first 36 eigenfaces.

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

%  Visualize the top 36 eigenvectors found
figure;
displayData(U(:, 1:36)');


% ================= Dimension Reduction for Faces ==================
%  Project images to the eigen space using the top k eigenvectors 

K = 100;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));


% ==== Visualization of Faces after PCA Dimension Reduction ====
%  Projected images to the eigen space using the top K eigen vectors and 
%  visualized only using those K dimensions
%  Original input is also displayed for comparison

K = 100;
X_rec  = recoverData(Z, U, K);
figure;
% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;
