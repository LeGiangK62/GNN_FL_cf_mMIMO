clear;
clc;
num_sam=2000; num_ap=30; num_ue=6; num_antenna=1;
tau=20; power_f=0.2; D=1; f=1900; Hb=15; Hm=1.65; d0=0.01;d1=0.05;

% fprintf('%d UE, %d AP', num_ue, num_ap);
% [betas, Phii_cf, R_cf_opt_min] = data_generation(num_sam, num_ap, num_ue, num_antenna, tau, power_f, Hb, Hm, f, d0, d1, D);
[betas, Gammas, Phii_cf, R_equal, R_frac, R_log, power] = downlink_sumrate_data(num_sam, num_ap, num_ue, num_antenna, tau, power_f, Hb, Hm, f, d0, d1, D);
                                                                    
filename = sprintf('dl_sumrate_data_%d_%d_%d.mat', num_sam, num_ue, num_ap);
save(filename,'betas', 'Gammas', 'Phii_cf', 'R_equal', 'R_frac', 'R_log', 'power');