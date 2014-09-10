function detector = dwot_get_avg_detector(renderer, azimuth, elevation, yaw, fov, model_indexes, model_class, param, bool_get_image)
if nargin < 9
  bool_get_image = true;
end

n_models = numel(model_indexes);


% Render the images and find optimal template size
im_model = cell(1,n_models);
im_size = zeros(n_models, 3);
for model_index = model_indexes
    % Assume that the model_indexes are base 1
    renderer.setModelIndex(model_index);
    
    % Assume that all models are viewpoint aligned
    renderer.setViewpoint(90-azimuth,elevation,yaw,0,fov);
    % model class and index are not supported yet
    im_model{model_index} = renderer.renderCrop();
    im_size(model_index, :) = size(im_model{model_index});
end


% Find optimal template size
%
%     1. Find largest image size and create template
max_image_size = max(im_size);

WHOTemplates = cell(1,n_models);

for model_index = 1:n_models
    total_padding_height = max_image_size(1) - im_size(model_index, 1);
    total_padding_width = max_image_size(2) - im_size(model_index, 2);
   
    x_offset = floor( total_padding_width / 2);
    y_offset = floor( total_padding_height / 2);

    padded_im = 255 * ones(max_image_size,'uint8');
    padded_im(y_offset+1:y_offset+im_size(model_index,1), x_offset+1:x_offset+im_size(model_index,2),:) = im_model{model_index};
    % Theoretically this should give the same size template size
    [ WHOTemplates{model_index}, ~, scale] = WHOTemplateCG_CUDA( padded_im, param);
    
    % save padded image
    im_model{model_index} = padded_im;
end

% Stupid
avg_whow = WHOTemplates{1};
if n_models > 1
    for model_index = 2:n_models
        avg_whow = avg_whow + WHOTemplates{model_index};
    end
end
avg_whow = avg_whow/n_models;

detector = [];
detector.whow = avg_whow;
detector.az = azimuth;
detector.el = elevation;
detector.yaw = yaw;
detector.fov = fov;
detector.model_indexes = model_indexes;
detector.sz = size(avg_whow);
padding = round(param.rendering_sbin / scale / 2);
detector.rendering_padding = padding;
if bool_get_image
  % Since the image requires averaging, we will use double precision format
  % average_image = ones(max_image_size);
  % average_image = padded_im
  % detector.rendering_image = paddedIm;
  average_image = single(im_model{1});
  if n_models > 1
      for model_index = 2:n_models
          average_image = average_image + single(im_model{model_index});
      end
  end
  
  % save it as rendering image of the detector.
  % The field exists for regular
  detector.rendering_image = average_image/n_models/255;
  
  % Also save individaul renderings of each model
  detector.rendering_images = im_model;
end
