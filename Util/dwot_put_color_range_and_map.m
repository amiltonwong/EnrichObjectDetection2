function param = dwot_put_color_range_and_map(param, n_range)

switch param.detection_mode
    case 'cnn'
        param.color_range = [ -inf -4 3 40 inf];
    case 'dwot'
        param.color_range = [-inf 20:5:100 inf];
    case 'vocdpm'
        param.color_range = [-inf -1:0.05:1 inf];
end

param.color_map = cool(numel(param.color_range));
