function detector = dwot_get_detector(azimuth, elevation, yaw, fov, model_index, model_class, param)
% model class and index are not supported yet
if ~isfield(param,'renderer')
	error('No renderer found');
end

param.renderer.setViewpoint(90-azimuth,elevation,yaw,0,fov);
im = param.renderer.renderCrop();

% [ WHOTemplate, HOGTemplate] = WHOTemplateDecompNonEmptyCell( im, Mu, Gamma, n_cell_limit, lambda, 50);
[ WHOTemplate, HOGTemplate] = WHOTemplateCG_CUDA( im, param);

detector = [];
detector.whow = WHOTemplate;
detector.az = azimuth;
detector.el = elevation;
detector.yaw = yaw;
detector.fov = fov;
detector.rendering_image = im;
detector.sz = size(WHOTemplate);