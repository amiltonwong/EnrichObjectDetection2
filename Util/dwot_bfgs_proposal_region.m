function [best_proposals]= dwot_bfgs_proposal_region(renderer, hog_region_pyramid, im_region, detectors, param, im, visualize)

if nargin <7 
    visualize = false;
end

if isfield(param,'bfgs_options')
    options = param.options;
else
%     options = optimoptions(@fminunc,'Algorithm','quasi-newton','MaxIter',5,'FinDiffRelStep',[0.001;0.001;0.001;0.001],'Display','none');
    
    options = optimoptions('fmincon','Algorith','sqp');
end

% org_cell_limit = param.n_cell_limit;
% param.n_cell_limit = 200;

n_proposal_region = numel(hog_region_pyramid);

n_batch = 1;
% org_cell_limit = param.n_cell_limit;
% param.n_cell_limit = 200;

best_proposals = cell(1, n_proposal_region);

for region_idx = 1:n_proposal_region
    template_idx = hog_region_pyramid{region_idx}.template_idx;
    best_state = struct('x', [detectors{template_idx}.az, detectors{template_idx}.el, detectors{template_idx}.yaw, detectors{template_idx}.fov],...
                    'models_idx', detectors{template_idx}.model_index,... % there can be multiple models and renderings 
                    'template_size', cell(1, n_batch),...
                    'rendering_image', detectors{template_idx}.rendering_image,...
                    'image_bbox', hog_region_pyramid{region_idx}.image_bbox,...
                    'score', hog_region_pyramid{region_idx}.det_score);
    
    current_state = struct('x', [detectors{template_idx}.az, detectors{template_idx}.el,...
                    detectors{template_idx}.yaw, detectors{template_idx}.fov],...
                    'models_idx', detectors{template_idx}.model_index,... % for multiple model
                    'score', hog_region_pyramid{region_idx}.det_score);
%   current_state = struct('az', detectors{template_idx}.az,...
%                   'el', detectors{template_idx}.el,...
%                   'yaw', detectors{template_idx}.yaw,...
%                   'fov', detectors{template_idx}.fov,...
%                   'models_idx', cell(1),...
%                   'energy', hog_region_pyramid{region_idx}.det_score);
    compute_score = @(x) -bfgs_detect_using_instant_detector(...
            renderer, hog_region_pyramid{region_idx},...
            x(1), x(2), x(3), x(4), current_state.models_idx, param, im_region{region_idx});
  
%     [X(:,region_idx), F(region_idx), exitflag] = fminunc(@(x)compute_score(x),...
%            [current_state.x(1), current_state.x(2), current_state.x(3), current_state.x(4)], options);
    lb = best_state.x - 5;
    lb(4) = max(lb(4),5)
    ub = lb + 10
    
    x0 = [current_state.x(1), current_state.x(2), current_state.x(3), current_state.x(4)];
    opts = optimoptions('fmincon','Algorithm','sqp','FinDiffRelStep',[1;1;1;1]);
    problem = createOptimProblem('fmincon','objective', ...
        @(x) compute_score(x),'x0',x0,'lb',lb,'ub',ub, ...
        'options',opts);
    gs = GlobalSearch('TolFun',0.1,'TolX',0.1,'StartPointsToRun','bounds-ineqs',...
                    'NumStageOnePoints',10,'NumTrialPoints',20);
    tic;
    [xgs,fval] = run(gs,problem);
    toc
    xgs
    fval
%     ms = MultiStart;
%     tic;
%     [xms,~,~,~,solsms] = run(ms,problem,15);
%     toc
%     xms
    % Call fmincon
%     [xlocal,fvallocal] = fmincon(problem)
    
%     [X(:,region_idx), F(region_idx), exitflag] = fminunc(@(x)compute_score(x),...
%                [current_state.x(1), current_state.x(2), current_state.x(3), current_state.x(4)], options);
    if 0
        figure(1); subplot(131);
        imagesc(rendering_image); axis equal; axis tight;
        subplot(132);
        imagesc(HOGpicture(HOGTemplate)); axis equal; axis tight;
        subplot(133);
        imagesc(HOGpicture(WHOTemplate)); axis equal; axis tight;
        disp('press any button to continue');
        waitforbuttonpress;
    end
    best_proposals{region_idx} = best_state;
end
% catch e
%   disp(e.message);
%   renderer.delete();
%   param.n_cell_limit = org_cell_limit;
%   rethrow(e);
% end
% param.n_cell_limit = org_cell_limit;


function max_score = bfgs_detect_using_instant_detector(renderer, hog_pyramid, az, el, yaw, fov, models_idx, param, im_region)
fprintf('%f %f %f %f\n',az, el, yaw, fov);
renderer.setViewpoint(az,el,yaw,0,fov);
rendering_image = renderer.renderCrop();
template = WHOTemplateCG_CUDA( rendering_image, param);

n_hog = numel(hog_pyramid.pyramid);
c = cell(1, n_hog);
max_score  = -inf;
max_level  = -1;
max_idx  = -1;
for level_idx = 1:n_hog
  % c{level_idx} = fconvblasfloat(hog_pyramid.pyramid(level_idx).padded_hog, {WHOTemplate}, 1, 1);
  c{level_idx} = dwot_conv(hog_pyramid.pyramid(level_idx).padded_hog, template, param);
  [max_c, max_c_idx] = max(c{level_idx}(:));
  if max_score < max_c;
    max_score = max_c;
    max_level = level_idx;
    max_idx   = max_c_idx;
  end
end

if max_level == -1
  return;
else
%   data_size = size(c{max_level});
%   [y_coord, x_coord] = ind2sub(data_size(1:2), max_idx);
%   y_coord = y_coord + hog_pyramid.pyramid(max_level).padded_hog_bbox(2) - 1;
%   x_coord = x_coord + hog_pyramid.pyramid(max_level).padded_hog_bbox(1) - 1;
% 
%   if param.computing_mode == 0
%     [y1, x1] = dwot_hog_to_img_conv(y_coord, x_coord, param.sbin, hog_pyramid.pyramid(max_level).scale, param.detect_pyramid_padding);
%     [y2, x2] = dwot_hog_to_img_conv(y_coord + template_size(1), x_coord + template_size(2), param.sbin, hog_pyramid.pyramid(max_level).scale, param.detect_pyramid_padding);
%   elseif param.computing_mode == 1
%     [y1, x1] = dwot_hog_to_img_fft(y_coord, x_coord, [1 1], param.sbin, hog_pyramid.pyramid(max_level).scale);
%     [y2, x2] = dwot_hog_to_img_fft(y_coord+ template_size(1),...
%                                     x_coord + template_size(2),...
%                                     [1 1], param.sbin, hog_pyramid.pyramid(max_level).scale);
%   else
%     error('Computing mode not supported');
%   end
%   image_bbox = [x1 y1, x2, y2];
  max_score = double(max_score);
  fprintf(['Score : ' num2str(max_score) '\n']);

  if 1
%     subplot(221);
%     imagesc(im_region);
  %   subplot(222);
  %   imagesc(HOGpicture(WHOTemplate));
%     subplot(223);
    imagesc(rendering_image);
    drawnow
  %   for level_idx = 1:n_hog
  %     subplot(224);
  %     imagesc(c{level_idx}{1});
  %     colorbar;
  %     waitforbuttonpress
  %   end
  end
end