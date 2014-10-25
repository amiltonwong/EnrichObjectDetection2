function success = dwot_analyze_and_visualize_cnn_results_unit_test()

success = false;

% fix the stupid test
load('/home/chrischoy/car_init_0_Car_each_27_lim_250_lam_0.1500_a_24_e_2_y_1_f_1.mat');
dwot_analyze_and_visualize_cnn_results( fullfile('Result/car_val', 'PASCAL12_car_val_init_0_Car_each_27_lim_250_lam_0.1500_a_24_e_2_y_1_f_1_scale_2.00_sbin_6_level_15_skp_n_server_104_cnn_proposal_dwot_tmp_12.txt') , detectors, '/home/chrischoy/DWOT_CNN5', VOCopts, param, skip_criteria, param.color_range, param.nms_threshold, false)

success = true;
