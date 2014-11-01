function [detection_result_file, detection_result_common_name] = dwot_get_detection_result_file_name(...
    DATA_SET, TEST_TYPE, SAVE_PATH, model_names, save_mode, server_id, param, appendix, b_make_file)
if nargin < 9
    b_make_file = false;
end

% make detection file name and make the file. 
[ detector_model_name ] = dwot_get_detector_model_name(model_names, param);

detection_result_file = sprintf(['%s_%s_%s_%s_lim_%d_lam_%0.3f_a_%d_e_%d_y_%d_f_%d_scale_',...
   '%0.2f_sbin_%d_level_%d_skp_%s_sm_%s_server_%s'],...
   DATA_SET, lower(param.class), TEST_TYPE, detector_model_name, param.n_cell_limit, param.lambda,...
   numel(param.azs), numel(param.els), numel(param.yaws), numel(param.fovs), param.image_scale_factor,...
   param.sbin, param.n_level_per_octave, param.skip_name, save_mode, server_id.num);

if nargin > 6 && ~isempty(appendix)
    detection_result_file = sprintf('%s_%s.txt', detection_result_file, appendix);
else
    detection_result_file = sprintf('%s.txt', detection_result_file);
end

% Check duplicate file name and return different name
if b_make_file
    detection_result_file = dwot_save_detection(SAVE_PATH, detection_result_file, true);
end

detection_result_common_name = regexp(detection_result_file, '\/?(?<name>.+)\.txt','names');
detection_result_common_name = detection_result_common_name.name;
