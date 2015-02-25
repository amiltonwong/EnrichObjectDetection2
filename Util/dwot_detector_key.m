function key = dwot_detector_key(azimuth, elevation, yaw, fov, model_indexes)
key = sprintf('%.2f_%.2f_%.2f_%.2f', azimuth, elevation, yaw, fov);
