function [label, image_path] = dwot_3d_object_dataset(DATA_PATH, CLASS)
% bboxFormat = 'x1 y1 x2 y2'

CLASS_PATH = [DATA_PATH '/' CLASS];

sub_dirs = dir(CLASS_PATH);
sub_dirs = sub_dirs(3:end);

image_path = {};
label = {};
image_index = 1;

if exist([CLASS_PATH '/annotation.mat'],'file')
  load([CLASS_PATH '/annotation.mat']);
  return;
end

% When they fil to find 
disp('Creating annotation files from scratch'); 
for dir_idx = 1:numel(sub_dirs)
    sub_dir = sub_dirs(dir_idx);
    sub_files = dir([CLASS_PATH '/' sub_dir.name]);
    sub_files = sub_files(3:end);
    
    for file_idx = 1:numel(sub_files)
        sub_file = sub_files(file_idx);
        [~, file_name, file_ext] = fileparts(sub_file.name);
        if isempty(strfind('.jpg.png.bmp',file_ext))
          continue;
        end
        
        image_path{image_index} = [CLASS_PATH '/' sub_dir.name '/' sub_file.name];
        
        FID = fopen([CLASS_PATH '/' sub_dir.name '/mask/' file_name '.mask']);
        mask = textscan(FID, '%d');
        fclose(FID); 
        maskSize = mask{1}(1:2);
        mask = mask{1}(3:end);
        mask = reshape(mask, maskSize');
    
        x_coord = find(sum(mask,1)>0);
        y_coord = find(sum(mask,2)>0);
        % bbox = [min(x_coord), min(y_coord), max(x_coord)-min(x_coord), max(y_coord)-min(y_coord)];
        bbox = [min(x_coord), min(y_coord), max(x_coord), max(y_coord)];
        
        label{image_index}.BB = bbox';
        label{image_index}.diff = false;
        label{image_index}.det = false;
        extracted_view =  regexp(sub_file.name,'A(?<az>\d+)_H(?<el>\d+)_S(?<dist>\d+)','names');
        label{image_index}.azimuth = ( str2num(extracted_view.az) - 1) * 45;
        label{image_index}.elevation = ( str2num(extracted_view.el) - 1) * 15;
        label{image_index}.distance = str2num( extracted_view.dist );
        image_index = image_index + 1;
    end
end

save([CLASS_PATH '/annotation.mat'], 'label', 'image_path');