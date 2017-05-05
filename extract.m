function [features] = extract( X, scale, filters)

window = [3,3];
overlap = [2,2];
border = [1,1];
O = zeros(1, scale - 1);
G = [1 O -1]; % Gradient
L = [1 O -2 O 1]/2; % Laplacian
conffilters = {G, G.', L, L.'}; % 2D versions
% Compute one grid for all filters
grid = sampling_grid(size(X), ...
    window, overlap, border, scale);
feature_size = prod(window) * numel(conffilters);

% Current image features extraction [feature x index]
if isempty(filters)
    f = X(grid);
    features = reshape(f, [size(f, 1) * size(f, 2) size(f, 3)]);
else
    features = zeros([feature_size size(grid, 3)], 'single');
    for i = 1:numel(filters)
        f = conv2(X, filters{i}, 'same');
        f = f(grid);
        f = reshape(f, [size(f, 1) * size(f, 2) size(f, 3)]);
        features((1:size(f, 1)) + (i-1)*size(f, 1), :) = f;
    end
end
