function dwot_draw_bounding_box(bbsNMS, param)
if ~isfield(param,'cnn_color_map')
    n_color = numel(param.cnn_color_range);
    param.cnn_color_map = jet(n_color);
end

for i = size(bbsNMS,1):-1:1
    bb = double(dwot_bbox_xy_to_wh(bbsNMS(i, 1:4)));
    [~, color_idx] = histc(bbsNMS(i,end), param.cnn_color_range);
    curr_color = param.cnn_color_map(color_idx, :);
    rectangle('position', bb,'edgecolor',[0.5 0.5 0.5],'LineWidth',3);
    rectangle('position', bb,'edgecolor',curr_color,'LineWidth',1);
    text(bb(1)+1 , bb(2), sprintf('s:%0.2f', bbsNMS(i,end)), ...
        'BackgroundColor',curr_color,'EdgeColor',[0.5 0.5 0.5],'VerticalAlignment','bottom');
end
axis equal;
axis tight;
axis off;