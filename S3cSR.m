function im_h = S3cSR(model, im_l)

%%
load(model);
param.L = 1024;
param.lambda = 0.05;
param.mode = 2;       % penalized formulation
param.approx=0;
scale = conf.scale;
V_pca = conf.V_pca;
dict_lores = conf.dict_lores;
M = conf.M;
dict_hores = conf.DICT_HIRES;
O = zeros(1, scale - 1);
G = [1 O -1]; % Gradient
L = [1 O -2 O 1]/2; % Laplacian
filters = {G, G.', L, L.'}; % 2D versions
im_b = imresize(im_l, scale, 'bicubic');
features = collect( {im_b}, scale, filters);
features = V_pca' * double(features);
coeffs = mexLasso(features, double(dict_lores),param);
coeffs = full(coeffs);
Z = coeffs ~= 0;
%% Reconstruct using patches' dictionary
coeffs = M*coeffs;
coeffs = Z.*coeffs;
patches = dict_hores * coeffs;

%% Add low frequencies to each reconstructed patch
patches = patches + collect({im_b}, scale, {});

%% Combine all patches into one image
%img_size = size(im_b) * conf.scale;
[nrow, ncol] = size(im_b);
grid = sampling_grid([nrow, ncol], ...
    [3,3], [2,2], [1,1], scale);
result = overlap_add(patches, [nrow, ncol], grid);

result = shave(result, [scale, scale]);
im_l = shave(double(im_l), [1, 1]);
%% Non-Local Means
% self-similarity
if scale == 2
    win_NLM = 3;
else
    win_NLM = 5;
end
N       =   Compute_NLM_Matrix( result , win_NLM);
NTN          =   N'*N*0.05;
im_f = sparse(double(result(:)));
for j = 1 : 30
    im_f = im_f  - NTN*im_f;
end
result = reshape(full(im_f), nrow-2*scale, ncol-2*scale);
maxIter = 20;
result = backprojection(result, im_l, maxIter);
im_h = result;