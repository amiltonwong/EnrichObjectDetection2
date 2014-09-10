% visualize and save model images
function dwot_visualize_models(renderer, model_files, azimuth, elevation, yaw, fov, save_path)

n_models = numel(model_files);

% Render the images and find optimal template size
im_model = cell(1,n_models);
for model_index = 1:n_models
    % Assume that the model_indexes are base 1
    renderer.setModelIndex(model_index);
    
    % Assume that all models are viewpoint aligned
    renderer.setViewpoint(90-azimuth,elevation,yaw,0,fov);
    % model class and index are not supported yet
    im_model{model_index} = renderer.renderCrop();
   
    imagesc(im_model{model_index}); axis off;
    title(strrep(model_files{model_index},'_',' '));
    waitforbuttonpress;
end

if nargin > 5
    % Save images to the path
    for model_index = 1:n_models 
        imagesc(im_model{model_index}); axis off;
        title(strrep(model_files{model_index},'_',' '));
       
        save_name = sprintf('%s/%s.png',...
                save_path, model_files{model_index});
        print('-dpng','-r100', save_name);
    end
end

