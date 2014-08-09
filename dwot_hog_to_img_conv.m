function [imgUIdx, imgVIdx] = dwot_hog_to_img_conv(hogUIdx, hogVIdx, sbin, scale, padder)
% convolutionMode = 0, default convolution using felzenswalb's fconvblas.
% In this case, it pad the image and convolve thus have to take padding
% into account.

imgUIdx = (hogUIdx - padder - 1) * sbin/scale + 1; % zero-base indexing to resize and add 1
imgVIdx = (hogVIdx - padder - 1) * sbin/scale + 1; % zero-base indexing to resize and add 1 