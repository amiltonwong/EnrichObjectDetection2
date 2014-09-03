az = 90;
el = 20;
yaw = 0;
fov = 25;

renderer.setViewpoint(az,el,yaw,0,fov);
im= renderer.render();


[bbsNMS, hog, scales] = dwot_detect( im, templates, param);

bbsNMS_clip = clip_to_image(bbsNMS, [1 1 imSz(2) imSz(1)]);
[bbsNMS_clip, tp{imgIdx}, fp{imgIdx}, ~] = dwot_compute_positives(bbsNMS_clip, gt(imgIdx), param);
bbsNMS(:,9) = bbsNMS_clip(:,9);
      
subplot(221); 
dwot_draw_overlap_detection(im, bbsNMS(:,:), renderings, n_proposals, 50, visualize_detection);

subplot(222);
imagesc(im); axis equal; axis off; 

subplot(223);
imagesc(HOGpicture(hog{bbsNMS(1,6)})); axis equal; axis off; 

subplot(224); 
imagesc(HOGpicture(detectors{bbsNMS(1,11)}.whow)); axis equal; axis off;