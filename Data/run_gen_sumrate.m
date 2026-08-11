clear;
clc;
num_sam=1000; num_ap=80; num_ue=25; num_antenna=1;
tau=25; power_f=0.2; D=1; f=1900; Hb=15; Hm=1.65; d0=0.01;d1=0.05;

% fprintf('%d UE, %d AP', num_ue, num_ap);
% [betas, Phii_cf, R_cf_opt_min] = data_generation(num_sam, num_ap, num_ue, num_antenna, tau, power_f, Hb, Hm, f, d0, d1, D);
[betas, Gammas, Phii_cf, R_equal, R_frac, R_log, power] = downlink_sumrate_data(num_sam, num_ap, num_ue, num_antenna, tau, power_f, Hb, Hm, f, d0, d1, D);
                                                                    
filename = sprintf('sumrate/adding_dl_sumrate_data_%d_%d_%d_%d.mat', num_sam, num_ue, num_ap, tau);
save(filename,'betas', 'Gammas', 'Phii_cf', 'R_equal', 'R_frac', 'R_log', 'power');



% 20 5
% 30 5
% 40 5


% 20 10
% 30 10
% 40 10


% 40 20 20 
% 60 20 20 
% 80 20 20 
% 40 25 25 

% 60 25 25 
% 80 25 25 
