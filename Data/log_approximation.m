function [alpha, beta, denom] = log_approximation(rho, Gammaan, BETAAn, PhiPhi)
    % Input: rho - the power matrix
    [num, denom] = dl_sinr_component_calculate(rho, Gammaan, BETAAn, PhiPhi);
    sinr = num./denom;
    alpha = sinr./(1+sinr);
    beta = log2(1+sinr) - alpha.*log2(sinr);
end