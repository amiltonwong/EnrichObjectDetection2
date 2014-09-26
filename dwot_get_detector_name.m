function [detector_model_name]= dwot_get_detector_name(CLASS, SUB_CLASS, model_names, param)

disp('Models to use : ');
nModels = numel(model_names);
for model_idx = 1:nModels
  fprintf('%d/%d : %s\n', model_idx, nModels, model_names{model_idx});
end


% detector name
detector_model_name = ['init_' num2str(param.template_initialization_mode) '_' CLASS];

if ~isempty(SUB_CLASS)
  detector_model_name = [ detector_model_name '_' SUB_CLASS ];
end


if nModels < 5
  detector_model_name = [ detector_model_name '_each_' strjoin(strrep(model_names, '/','_'),'_')];
else
  detector_model_name = [ detector_model_name '_each_' num2str(nModels)];
end
