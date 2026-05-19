function kappa = crlb_linear_check(power, b, A, nu)
    p_sen = sum(power, 2);              % M×1 vector
    crlb = b - nu * A * p_sen;        % d×1 vector
    crlb_violation = max(crlb);            % Scalar violation

    if crlb_violation > 0
        % Compute scaling factor κ
        A_p_sen = A * p_sen;
        kappa = max(b ./ (nu * A_p_sen + eps));  % eps to avoid division by zero
        kappa = max(kappa, 1);  % Ensure κ ≥ 1
    else
        kappa = 1;
    end
end