function [detector_name, detector_file_name]= dwot_get_detector_name(model_names, param)
[ detector_model_name ] = dwot_get_detector_model_name(model_names, param);

detector_name = sprintf('%s_%s_lim_%d_lam_%0.3f_a_%d_e_%d_y_%d_f_%d',...
        lower(param.class),  detector_model_name, param.n_cell_limit, param.lambda,...
        numel(param.azs), numel(param.els), numel(param.yaws), numel(param.fovs));
    
detector_file_name = sprintf('%s.mat', detector_name);