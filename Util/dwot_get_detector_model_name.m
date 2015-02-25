function [detector_model_name]= dwot_get_detector_model_name( model_names, param)


nModels = numel(model_names);
% disp('Models to use : ');
% for model_idx = 1:nModels
%   fprintf('%d/%d : %s\n', model_idx, nModels, model_names{model_idx});
% end


% detector name
detector_model_name = ['init_' num2str(param.template_initialization_mode) '_' lower(param.class)];

if ~isempty(param.sub_class)
  detector_model_name = [ detector_model_name '_' lower(param.sub_class) ];
end

if nModels < 3
  detector_model_name = [ detector_model_name '_each_' strjoin(strrep(model_names, '/','_'),'_')];
else
  detector_model_name = [ detector_model_name '_each_' num2str(nModels)];
end
