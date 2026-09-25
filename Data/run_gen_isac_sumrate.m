clear;
clc;
num_sam=2000; num_ap=30; num_ue=6; num_antenna=1; num_sr=2; nu=1; 
tau=20; power_f=0.2; D=1; f=1900; Hb=15; Hm=1.65; d0=0.01;d1=0.05;

[betas, Gammas, Phii_cf, R_equal, R_frac, R_log, ...
    power_eq, power_frac, power_log, ...
    rcs_values, ap_locations, sr_locations,...
    q_a_all, q_b_all, q_c_all] = downlink_ISAC_sumrate_data(num_sam, num_ap, num_ue, num_sr, num_antenna, tau, power_f, Hb, Hm, f, d0, d1, D, nu);
                                                                    
filename = sprintf('New/dl_isac_sumrate_data_%d_%d_%d.mat', num_sam, num_ue, num_ap);
save(filename,'betas', 'Gammas', 'Phii_cf', 'R_equal', 'R_frac', 'R_log', ...
    'rcs_values', 'ap_locations', 'sr_locations', 'q_a_all', 'q_b_all', 'q_c_all', ...
    'power_eq', 'power_frac', 'power_log');


% 30 6
% 50 6
%100 6
% 30 10
% 50 10
%100 10
% 50 15
%100 15
% 30 4
% 30 8
% 30 15
% 30 16
% 30 12
% 40 6
% 60 6
% 80 6
% 20 6










