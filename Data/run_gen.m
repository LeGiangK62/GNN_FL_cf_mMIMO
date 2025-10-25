
clear;
num_sam=500; num_ap=30; num_ue=6;
tau=20; power_f=0.2; D=1; f=1900; Hb=15; Hm=1.65; d0=0.01;d1=0.05;

% fprintf('%d UE, %d AP', num_ue, num_ap);
[betas, Phii_cf, R_cf_opt_min] = data_generation(num_sam, num_ap, num_ue, tau, power_f, Hb, Hm, f, d0, d1, D);

filename = sprintf('eval_data_%d_%d_%d.mat', num_sam, num_ue, num_ap);
save(filename,'betas','Phii_cf','R_cf_opt_min');