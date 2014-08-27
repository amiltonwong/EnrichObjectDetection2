function detectors = dwot_find_detector(detectors_bulk, detector_table, azimuth, elevation, yaw, fov, model_index, model_class, param)
% azs  = param.azs;
% els  = param.els;
% yaws = param.yaws;
% fovs = param.fovs;
% model_indexes = param.model_indexes;
% model_class   = param.model_class;

% az_idx = find(azs == azimuth);
% el_idx = find(els == elevation);
% yaw_idx = find(yaws == yaw);
% fov_idx = find(fovs == fov);
% model_idx = find(model_indexes == model_idx);

% detector = [];
% if isempty(az_idx) || isempty(el_idx) || isempty(yaw_idx) || isempty(fov_idx) % || isempty(model_idx)
% 	return;
% end

% n_azs = param.n_azs;
% n_els = param.n_els;
% n_yaws = param.n_yaws;
% n_fovs = param.n_fovs;

n_query 	 = numel(azimuth);
template_idx = zeros(1, n_query);
detectors 	 = cell(1, n_query);
for idx = 1:n_query
	az = mod(azimuth(idx),360);
	
	key = dwot_detector_key(az, elevation(idx), yaw(idx), fov(idx));
	if detector_table.isKey(key)
		template_idx(idx) = detector_table( key );
		detectors{idx} = detectors_bulk{template_idx};
	else
		detectors{idx} = dwot_get_detector(az, elevation(idx), yaw(idx), fov(idx), [1], 'not_supported_model_class', param);
	end
end