function new_file_name = dwot_save_detection(save_path, file_name, b_new_file, save_format, structure_data, image_name)

if ~exist('b_new_file','var')
    b_new_file = false;
end

if ~b_new_file && ~exist('save_format','var')
    error('save format undefined');
end

% If we have to create new file, make one. Also if there is existing 
% file name, make file name appended with temporary number
if b_new_file
    [f, new_file_name] = get_new_file(save_path, file_name);
else
    f = fopen(fullfile(save_path, file_name),'a');
end

% save results to file
if exist('structure_data','var') && ~isempty(structure_data)
    dwot_save_load_delegate(f, save_format, image_name, structure_data);
end

% close file
fclose(f);



function [fid, file_name] = get_new_file(save_path, file_name)
file_list = dir(save_path);
file_list = file_list(~[file_list.isdir]);

if exist(fullfile(save_path,file_name),'file')
  [~,file_name_wo_ext] = fileparts(file_name);
  
  temp_numbers = cellfun(@(x) regexp(x, [file_name_wo_ext '_tmp_(?<num>\d+)\.txt'],'names'),...
                         {file_list.name},'UniformOutput',false);
  temp_numbers = temp_numbers(cellfun(@(x) ~isempty(x), temp_numbers));
  temp_numbers = cellfun(@(x) str2double(x.num), temp_numbers);
  if isempty(temp_numbers)
    curr_temp_idx = 1;
  else
    curr_temp_idx = max(temp_numbers) + 1;
  end
  [~, file_name_wo_ext, ext] = fileparts(file_name);
  file_name = sprintf('%s_tmp_%d%s',file_name_wo_ext,curr_temp_idx, ext);
  fprintf('file name exists, appending temporary number %d\n', curr_temp_idx);
end
fid = fopen(fullfile(save_path, file_name),'w+');
