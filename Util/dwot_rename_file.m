function [ file_names, file_paths ]= dwot_rename_file(PATH, pattern, replace , extension )
    % Directory path can be arbitrary deep for each sub classes.
    % Get all possible sub-classes

    [ file_names, file_paths ] = recursive_strrep(PATH, {}, {}, pattern, replace, extension);
%    detector_model_name = ['each_' strjoin(strrep(CAD_ROOT_DIR, '/','_'),'_')];
%    model_files = cellfun(@(x) [model_paths strrep([x '.3ds'], '/', '_')], model_names, 'UniformOutput', false);
end
   
function [ file_names, file_paths ] = recursive_strrep(PATH, file_names, file_paths, pattern, replace, extension)
    lists = dir(PATH);
    directories = lists([lists.isdir]);
    % the default directories are . and ..
    nDir = numel(directories);
    if nDir > 2
        for dir_idx = 3:nDir
            [ file_names, file_paths ] = recursive_strrep(fullfile(PATH, directories(dir_idx).name), file_names, file_paths, pattern, replace, extension);
        end
    end
    
    files = lists(~[lists.isdir]);
    for file_idx = 1:numel(files)
        file = files(file_idx); 
        substrs = regexp(file.name, '^(?<name>[a-zA-Z0-9-_\.]+)\.(?<ext>\w{3})$','names');
        if isempty(substrs) || ~strcmp(extension, substrs.ext ) 
          continue;
        end

        new_name = strrep(substrs.name, pattern, replace);
        system(['mv  ' fullfile(PATH, file.name) ' ' fullfile(PATH, [new_name '.' extension]) ]); 
        % system(['unzip ' fullfile(PATH,file.name) ' -d ' fullfile(PATH,substrs.name)]);
        % system(['osgconv ' fullfile(PATH,file.name) ' ' fullfile(PATH,substrs.name) '.3ds']);
        file_names = { file_names{:}, new_name};
        file_paths = { file_paths{:}  fullfile(PATH,[new_name '.' extension]) };
    end
end

