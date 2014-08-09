function [hogUIdx, hogVIdx] = dwot_img_to_hog_fft(imgUIdx, imgVIdx, templateSize, sbin, scale)

hogUIdx = ( imgUIdx - 1 ) * scale / sbin + templateSize(1);
hogVIdx = ( imgVIdx - 1 ) * scale / sbin + templateSize(2);
