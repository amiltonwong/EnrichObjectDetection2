function new_file_name = dwot_save_detection(detection_result, save_path, file_name, image_name, b_new_file, save_mode)

if ~exist('b_new_file','var')
  b_new_file = false;
end

if ~exist('save_mode','var')
  save_mode = 0;
end

%% 
% If we have to create new file, make one. Also if there is existing 
% file name, make file name appended with temporary number
if b_new_file
  file_list = dir(save_path);
  file_list = file_list(~[file_list.isdir]);
  
  if exist(fullfile(save_path,file_name),'file')
    warning('file name exists, appending temporary number');
    [~,file_name_wo_ext] = fileparts(file_name);
    
    temp_numbers = cellfun(@(x) regexp(x, [file_name_wo_ext '_tmp_(?<num>\d+)\.txt'],'names'),{file_list.name},'UniformOutput',false);
    temp_numbers = temp_numbers(cellfun(@(x) ~isempty(x), temp_numbers));
    temp_numbers = cellfun(@(x) str2double(x.num), temp_numbers);
    if isempty(temp_numbers)
      curr_temp_idx = 1;
    else
      curr_temp_idx = max(temp_numbers) + 1;
    end
    [~, file_name_wo_ext, ext] = fileparts(file_name);
    file_name = sprintf('%s_tmp_%d%s',file_name_wo_ext,curr_temp_idx, ext);
  end
  f = fopen(fullfile(save_path, file_name),'w+');
else
  f = fopen(fullfile(save_path, file_name),'a');
end

new_file_name = file_name;

n_detection = size(detection_result,1);
for det_idx = 1:n_detection
  % The read detection uses the following line to read data
  % [ids,conf,x1,y1,x2,y2]=textread(detfn,'%s %f %f %f %f %f');
  if save_mode == 0 
    fwrite(f,sprintf('%s %f %f %f %f %f\n',...
                      image_name,...
                      detection_result(det_idx, end),...
                      detection_result(det_idx, 1:4)));
  else
    fwrite(f,sprintf('%s %f %f %f %f %f %d\n',...
                      image_name,...
                      detection_result(det_idx, end),... % detection score
                      detection_result(det_idx, 1:4),... % bbox
                      detection_result(det_idx, 11)));  % templateIdx
  end
end
fclose(f);
