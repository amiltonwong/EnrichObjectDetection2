n_mcmc = 1;

      
param.detect_min_scale             = 0.2;
param.detect_max_scale            =  0.5;
param.detect_levels_per_octave = 50;

for detector_idx = 1:50:780
    im = padarray(detectors{detector_idx}.rendering_image,[50, 50, 0],255);
    if COMPUTING_MODE == 0
      [bbsAllLevel, hog, scales] = dwot_detect( im, templates_cpu, param);
%       [hog_region_pyramid, im_region] = dwot_extract_region_conv(im, hog, scales, bbsNMS, param);
%       [bbsNMS_MCMC] = dwot_mcmc_proposal_region(im, hog, scale, hog_region_pyramid, param);
    elseif COMPUTING_MODE == 1
      % [bbsNMS ] = dwot_detect_gpu_and_cpu( im, templates, templates_cpu, param);
      [bbsAllLevel, hog, scales] = dwot_detect_gpu( im, templates_gpu, param);
%       [hog_region_pyramid, im_region] = dwot_extract_region_fft(im, hog, scales, bbsNMS, param);
    elseif COMPUTING_MODE == 2
      [bbsAllLevel, hog, scales] = dwot_detect_combined( im, templates_gpu, templates_cpu, param);
    else
      error('Computing Mode Undefined');
    end
    
    bbsNMS = esvm_nms(bbsAllLevel, param.nms_threshold);

    [hog_region_pyramid, im_region] = dwot_extract_hog(hog, scales, detectors, bbsNMS(1:n_mcmc,:), param, im, 1);
    subplot(224);
    imagesc(detectors{bbsNMS(1,11)}.rendering_image);
    axis equal; axis tight;
  
    dwot_draw_overlap_detection(im, bbsNMS(1,:), renderings, inf, 50, visualize_detection, [0.2, 0.8, 0.0], color_range );

    waitforbuttonpress;
end