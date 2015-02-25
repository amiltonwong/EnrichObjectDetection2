function success = dwot_analyze_and_visualize_3D_object_results_unit_test()

success = false;

% fix the stupid test
load('/home/chrischoy/car_init_0_Car_Sedan_each_9_lim_250_lam_0.150_a_24_e_4_y_1_f_2.mat');
dwot_analyze_and_visualize_3D_object_results(fullfile('Result','3DObject_car_init_0_Car_Sedan_each_9_lim_250_lam_0.150_a_24_e_4_y_1_f_2_scale_1.00_sbin_4_level_15_server_102_tmp_1.txt'), ...
    detectors, '/home/chrischoy/DWOT_3D_CAR', param, DATA_PATH, CLASS, param.color_range, 0.3, false, 1, 180)

success = true;
