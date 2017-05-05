close all;
clear all;
addpath('SPAMS');
addpath('SPAMS/build');
%% read ground truth image
im  = imread('Set5\bird.bmp');
%im  = imread('Set14\foreman.bmp');

%% set parameters
up_scale = 4;
model = ['model\x' num2str(up_scale) '.mat'];

%% work on illuminance only
if size(im,3)>1
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end

im_gnd = modcrop(im, up_scale);
im_gnd = single(im_gnd)/255;

%% bicubic interpolation
im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
im_b = imresize(im_l, up_scale, 'bicubic');

%% S3cSR
im_h = S3cSR(model, im_l);

%% remove border
im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);
im_h = uint8(im_h * 255);
%% compute PSNR
psnr_bic = compute_psnr(im_gnd,im_b);
psnr_S3cSR = compute_psnr(im_gnd,im_h);

%% show results
fprintf('PSNR for Bicubic Interpolation: %.2f dB\n', psnr_bic);
fprintf('PSNR for S3cSR Reconstruction: %.2f dB\n', psnr_S3cSR);

figure, imshow(im_b); title('Bicubic Interpolation');
figure, imshow(im_h); title('S3cSR Reconstruction');

%imwrite(im_b, ['Bicubic Interpolation' '.bmp']);
%imwrite(im_h, ['S3cSR Reconstruction' '.bmp']);
