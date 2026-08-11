function R_sum = dl_rate_calculate(rho, Gammaan, BETAAn, PhiPhi)
% Compute DL sum-rate for power allocation matrix rho (M x K).

    % rho     = gpuArray(rho);
    % Gammaan = gpuArray(Gammaan);
    % BETAAn  = gpuArray(BETAAn);
    % PhiPhi  = gpuArray(PhiPhi);

    sinr = dl_sinr_calculate(rho, Gammaan, BETAAn, PhiPhi);
    all_rate = log2(1+sinr);
    
    R_sum = sum(all_rate);
    
end