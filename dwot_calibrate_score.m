function calibrated_score = dwot_calibrate_score(score, detector_idx, detectors, param)
switch param.calibration_mode
    case 'gaussian'
        calibrated_score = (score - detectors{detector_idx}.mean)/detectors{detector_idx}.sigma ;            
    case 'linear'
        calibrated_score = detectors{detector_idx}.a * score + detectors{detector_idx}.b;
    otherwise
        error(['Calibration mode undefined : ', param.calibration_mode]);
end