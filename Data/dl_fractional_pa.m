function rho = dl_fractional_pa(BETAA, M, K, P_max, theta)
% Fractional PA: each AP m distributes P_max among UEs proportional to
% beta_mk^theta.  Stronger large-scale channel -> more power.
%
%   rho_mk = P_max * beta_mk^theta / sum_{k'} beta_mk'^theta
%
    rho = zeros(M, K);
    for m = 1:M
        w     = BETAA(m,:) .^ theta;
        w_sum = sum(w);
        if w_sum < 1e-15; w_sum = 1e-15; end
        rho(m,:) = P_max * w / w_sum;
    end
end