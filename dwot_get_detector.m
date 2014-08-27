function detector = dwot_get_detector(renderer, azimuth, elevation, yaw, fov, model_index, model_class, param)
% model class and index are not supported yet
renderer.setViewpoint(90-azimuth,elevation,yaw,0,fov);
im = param.renderer.renderCrop();

% [ WHOTemplate, HOGTemplate] = WHOTemplateDecompNonEmptyCell( im, Mu, Gamma, n_cell_limit, lambda, 50);
[ WHOTemplate] = WHOTemplateCG_CUDA( im, param);

detector = [];
detector.whow = WHOTemplate;
detector.az = azimuth;
detector.el = elevation;
detector.yaw = yaw;
detector.fov = fov;
detector.rendering_image = im;
detector.sz = size(WHOTemplate);