n_mcmc = 1;

for detector_idx = 1:50:780
  im = detectors{detector_idx}.rendering_image;
  [bbsAllLevel, hog, scales] = dwot_detect_gpu( im, templates_gpu, param);
  bbsNMS = esvm_nms(bbsAllLevel, param.nms_threshold);

  [hog_region_pyramid, im_region] = dwot_extract_hog(hog, scales, detectors, bbsNMS(1:n_mcmc,:), param, im, 1);
  subplot(224);
  imagesc(detectors{bbsNMS(1,11)}.rendering_image);
  axis equal; axis tight;
  waitforbuttonpress;
end