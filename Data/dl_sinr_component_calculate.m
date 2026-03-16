function [num, denom] = dl_sinr_component_calculate(rho, Gammaan, BETAAn, PhiPhi)
% Compute DL sum-rate for power allocation matrix rho (M x K).
    
    [M,K] = size(Gammaan);
    
    num = sum(sqrt(rho).*Gammaan,1).^2; % 1, K
    % denom = zeros(1,K);
    % 
    % for k = 1:K
    %     PC = 0; UI = 0;
    %     for k_prime = 1:K
    %         for m = 1:M
    %             PC = PC + rho(m,k_prime)*Gammaan(m,k_prime)*BETAAn(m,k);
    %         end
    % 
    %         if k ~= k_prime
    %             tmp = 0;
    %             for m = 1:M
    %                 tmp = tmp + sqrt(rho(m,k_prime)) * Gammaan(m,k_prime) * BETAAn(m,k) / BETAAn(m,k_prime);
    %             end
    %             UI = UI + tmp^2*PhiPhi(k,k_prime);
    %         end
    %     end
    %     denom(k) = 1 + PC + UI;
    % end

    PC = sum(BETAAn' * (rho .* Gammaan), 2)';
    V     = sqrt(rho) .* Gammaan ./ BETAAn;
    inner = BETAAn' * V;                       
    inner_sq = inner.^2;                       
    inner_sq  = inner_sq .* (1 - eye(K));
    UI = sum(inner_sq .* PhiPhi, 2)';  
    denom = 1 + PC + UI;
    
    % sinr = num./denom;
end