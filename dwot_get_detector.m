function detector = dwot_get_detector(renderer, azimuth, elevation, yaw, fov, model_index, model_class, param, bool_get_image)
if nargin < 9
  bool_get_image = true;
end
% model class and index are not supported yet
renderer.setViewpoint(azimuth,elevation,yaw,0,fov);
[im, depth] = renderer.renderCrop();
if isempty(im)
  error('Rendering error');
end
% TODO replace it
% 0 NZ-WHO
% 1 Constant # active cell in NZ-WHO
% 2 Decorrelate all but center only the non-zero cells
% 3 NZ-WHO but normalize by # of active cells

% 4 HOG feature
% 5 Whiten all, WHO-CG
% 6 Whiten all, WHO-CG-ZZ
% 7 center non zero, whiten all, zero out empty, NZC-WHO-CG-ZZ
% 8 Similar to 7 but find bias heuristically
% 9 Decomposition, NZC-WHO-Chol-ZZ
% 10 Decomposition WHO-Chol
% 11 NZ-WHO-Chol
if param.template_initialization_mode == 4
    [ WHOTemplate, scale] = HOGTemplate(im, param);
elseif param.template_initialization_mode == 9
    [ WHOTemplate, scale] = WHOTemplateDecomp(im, param);
elseif param.template_initialization_mode == 10
    [ WHOTemplate, scale] = WHOTemplateDecompStandard(im, param);
else
    [ WHOTemplate, ~, scale] = WHOTemplateCG_CUDA_various_whitening( im, param);
end

detector = [];
detector.whow = WHOTemplate;
detector.az = azimuth;
detector.el = elevation;
detector.yaw = yaw;
detector.fov = fov;
detector.sz = size(WHOTemplate);
padding = round(param.rendering_sbin / scale / 2);
detector.rendering_padding = padding;
detector.model_index = model_index;

if bool_get_image
  size_im = size(im);
  paddedIm = 255 * ones(size_im + [2 * padding, 2 * padding, 0],'uint8');
  paddedIm(padding+1:padding+size_im(1), padding+1:padding+size_im(2),:) = im;
  detector.rendering_image = paddedIm;
  
  paddedDepth = zeros(size_im(1:2) + [2 * padding, 2 * padding],'double');
  paddedDepth(padding+1:padding+size_im(1), padding+1:padding+size_im(2)) = depth;
  detector.rendering_depth = paddedDepth;
end
