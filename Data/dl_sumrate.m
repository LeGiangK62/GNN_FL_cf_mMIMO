function R_sum = dl_sumrate(rho, Gammaan, UI, K, M)
% Compute DL sum-rate for power allocation matrix rho (M x K).
%
% SINR_k = (sum_m rho_mk * gamma_mk)^2
%          / (sum_m rho_mk*gamma_mk  +  sum_{k2~=k} UI(k,k2)*sum_m(rho_mk2*gamma_mk2)  +  1)
%
% UI(k,k2) here is a scalar coefficient; the per-AP weighting for the
% interference term is sum_m rho_mk2 * gamma_mk2 (same form as signal).
 
    R_sum = 0;
    % Pre-compute weighted sums S(k) = sum_m rho_mk * gamma_mk  for all k
    S = zeros(K,1);
    for k = 1:K
        S(k) = sum(rho(:,k) .* Gammaan(:,k));
    end
    for k = 1:K
        sig   = S(k)^2;
        bu    = S(k);
        iu    = 0;
        for k2 = 1:K
            if k2 ~= k
                iu = iu + UI(k,k2) * S(k2);
            end
        end
        denom = bu + iu + 1;
        R_sum = R_sum + log2(1 + sig / denom);
    end
end