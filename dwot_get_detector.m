function detector = dwot_get_detector(renderer, azimuth, elevation, yaw, fov, model_index, model_class, param, bool_get_image)
if nargin < 9
  bool_get_image = true;
end
% model class and index are not supported yet
renderer.setViewpoint(90-azimuth,elevation,yaw,0,fov);
im = renderer.renderCrop();

% [ WHOTemplate, HOGTemplate] = WHOTemplateDecompNonEmptyCell( im, Mu, Gamma, n_cell_limit, lambda, 50);
[ WHOTemplate, ~, scale] = WHOTemplateCG_CUDA( im, param);

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
end