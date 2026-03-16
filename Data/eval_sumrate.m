function R = eval_sumrate(rho, Gammaan, UI, K)
    S = zeros(K,1);
    for k = 1:K
        S(k) = sum(rho(:,k) .* Gammaan(:,k));
    end
    R = 0;
    for k = 1:K
        iu = 0;
        for k2 = 1:K
            if k2 ~= k; iu = iu + UI(k,k2)*S(k2); end
        end
        R = R + log2(1 + S(k)^2 / (S(k) + iu + 1));
    end
end