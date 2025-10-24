function gen_data_with_input(num_sam, num_ap, num_ue, tau, power_f, D, f, Hb, Hm, d0, d1)

    % fprintf('%d UE, %d AP', num_ue, num_ap);
    [betas, Phii_cf, R_cf_opt_min] = data_generation(num_sam, num_ap, num_ue, tau, power_f, Hb, Hm, f, d0, d1, D);

    filename = sprintf('cf_data_%d_%d_%d.mat', num_sam, num_ue, num_ap);
    save(filename,'betas','Phii_cf','R_cf_opt_min');
