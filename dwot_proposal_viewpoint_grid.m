function [azs, els, yaws, fovs] = dwot_proposal_viewpoint_grid(az,  el,  yaw,  fov,...
                                                      daz, del, dyaw, dfov,...
                                                      naz, nel, nyaw, nfov, param)

n_az_views = 2 * naz + 1;
n_el_views = 2 * nel + 1;
n_yaw_views = 2 * nyaw + 1;
n_fov_views = 2 * nfov + 1;

fovs = (-nfov:nfov)' * dfov + fov;
fovs = repmat(fovs, [n_az_views * n_el_views * n_yaw_views, 1]);
fovs = fovs(:);

yaws = (-nyaw:nyaw)' * dyaw + yaw;
yaws = repmat(yaws, [n_az_views * n_el_views, n_fov_views])';
yaws = yaws(:);

els = (-nel:nel)' * del + el;
els = repmat(els, [n_az_views, n_yaw_views * n_fov_views])';
els = els(:);

azs = mod((-naz:naz)' * daz + az, 360);
azs = repmat(azs, [1, n_el_views * n_yaw_views * n_fov_views])';
azs = azs(:);