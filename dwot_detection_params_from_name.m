function detection_params = dwot_detection_params_from_name(name)

% First find dataset name and class
detection_params = regexp(name,...
  ['\/?(?<DATA_SET>[a-zA-Z0-9]+)_((half)?pad_detection_)?'...
  '(?<LOWER_CASE_CLASS>[a-zA-Z]+)_(?<TYPE>[a-zA-Z]+)_(?<detector_model_name>[\w_-]+)_lim'],'names');


detection_params_temp = regexp(name,...
  ['lim_(?<n_cell_limit>\d+)_lam_(?<lambda>[\d.]+)_a_(?<n_az>\d+)_e_(?<n_el>\d)_y_(?<n_yaw>\d+)_',...
  'f_(?<n_fov>\d+)'],'names');

detection_params = add_fields(detection_params, detection_params_temp);

detection_params_temp = regexp(name,...
   'scale_(?<scale>[\d.]+)_','names');
detection_params = add_fields(detection_params, detection_params_temp);
detection_params.scale = str2num(detection_params.scale);

detection_params_temp = regexp(name,...
  'sbin_(?<sbin>\d+)','names');

detection_params = add_fields(detection_params, detection_params_temp);

detection_params_temp = regexp(name,...
  'level_(?<n_level>\d+)_','names');

detection_params = add_fields(detection_params, detection_params_temp);

detection_params_temp = regexp(name,...
  'nms_(?<nms_threshold>[\d.]+)','names');

detection_params = add_fields(detection_params, detection_params_temp);

detection_params_temp = regexp(name,...
  'skp_(?<skip_name>[a-z]+)_','names');

detection_params = add_fields(detection_params, detection_params_temp);

detection_params_temp = regexp(name,...
  'sm_(?<save_mode>[a-z]+)_','names');

detection_params = add_fields(detection_params, detection_params_temp);

function params = add_fields(params, param_add)
new_fields = fields(param_add);
for new_field  = new_fields'
    new_field = new_field{1};
    if isempty(param_add)
        continue;
    end
    params.(new_field) = param_add.(new_field);
end


