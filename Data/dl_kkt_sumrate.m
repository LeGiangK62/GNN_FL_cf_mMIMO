function rho_opt = dl_kkt_sumrate(Gammaan, BETAAn, PhiPhi, P_max)
% =========================================================================
% DL cf-mMIMO Sum-Rate Maximisation via SCA (KKT-based)
%
% Matches dl_rate_calculate exactly:
%   num(k)   = ( sum_m sqrt(rho_mk) * gamma_mk )^2
%   PC(k)    = sum_{k'} sum_m rho_mk' * gamma_mk' * beta_mk      [all k']
%   UI(k)    = sum_{k'~=k} PhiPhi(k,k')
%              * ( sum_m sqrt(rho_mk') * gamma_mk' * beta_mk / beta_mk' )^2
%   denom(k) = 1 + PC(k) + UI(k)
%   SINR(k)  = num(k) / denom(k)
%
% Change of variable:  x_mk = sqrt(rho_mk)  =>  rho_mk = x_mk^2
%   num(k)   = ( sum_m x_mk * gamma_mk )^2                [quadratic in x(:,k)]
%   PC(k)    = sum_{k'} sum_m x_mk'^2 * gamma_mk'*beta_mk [convex quadratic]
%   UI(k)    = sum_{k'~=k} PhiPhi(k,k')
%              * ( sum_m x_mk' * gamma_mk'*beta_mk/beta_mk' )^2
%
% SCA surrogate for log2(1+SINR_k) around iterate x^(t):
%   S_k(x) = sum_m x_mk * gamma_mk   is LINEAR in x(:,k)
%   num_k(x) = S_k(x)^2  lower-bounded by: 2*S_k^t*S_k(x) - (S_k^t)^2
%   Denominator fixed at den_k^t  (concave lower bound on SINR_k)
%   After further linearising log2(1+.):
%     f_k(x) >= A_k * S_k(x) + C_k
%   where:
%     A_k = (2/ln2) * S_k^t / ( den_k^t * (1 + SINR_k^t) )
%     C_k = log2(1+SINR_k^t) - A_k * S_k^t
%
% Each CVX subproblem is an LP with SOCP constraint sum_k x_mk^2 <= P_max.
%
% Inputs:
%   Gammaan  (M x K)  MMSE gain normalised by Pu
%   BETAAn   (M x K)  large-scale fading normalised by Pu
%   PhiPhi   (K x K)  |phi_k^H phi_k'|^2
%   P_max    scalar   per-AP power budget on rho  (x_mk^2 budget per row)
%
% Output:
%   rho_opt  (M x K)  optimised power allocation
% =========================================================================

[M, K] = size(Gammaan);
max_iter = 60;
tol      = 1e-5;

% ---- Initialise with equal PA ----
x_cur   = sqrt(P_max / K) * ones(M, K);   % x_mk = sqrt(rho_mk)
rho_opt = [];

R_old = dl_rate_calculate(x_cur.^2, Gammaan, BETAAn, PhiPhi);
% fprintf('  KKT init:  R_sum = %.4f\n', R_old);

for iter = 1:max_iter

    % ================================================================
    % Step 1: evaluate building blocks at current x^t
    % ================================================================

    % S_k^t = sum_m x_mk^t * gamma_mk   (K x 1)
    S_t = zeros(K, 1);
    for k = 1:K
        S_t(k) = sum(x_cur(:,k) .* Gammaan(:,k));
    end

    % PC_k^t = sum_{k'} sum_m x_mk'^t^2 * gamma_mk' * beta_mk   (K x 1)
    PC_t = zeros(K, 1);
    for k = 1:K
        for kp = 1:K
            PC_t(k) = PC_t(k) + sum(x_cur(:,kp).^2 .* Gammaan(:,kp) .* BETAAn(:,k));
        end
    end

    % UI_k^t = sum_{k'~=k} PhiPhi(k,k')
    %          * ( sum_m x_mk'^t * gamma_mk' * beta_mk / beta_mk' )^2   (K x 1)
    UI_t = zeros(K, 1);
    for k = 1:K
        for kp = 1:K
            if kp ~= k
                v = sum(x_cur(:,kp) .* Gammaan(:,kp) .* BETAAn(:,k) ./ BETAAn(:,kp));
                UI_t(k) = UI_t(k) + PhiPhi(k,kp) * v^2;
            end
        end
    end

    den_t  = 1 + PC_t + UI_t;          % K x 1
    SINR_t = S_t.^2 ./ den_t;          % K x 1

    % ================================================================
    % Step 2: surrogate coefficients
    % ================================================================
    A_k = (2 / log(2)) * S_t ./ (den_t .* (1 + SINR_t));   % K x 1
    C_k = log2(1 + SINR_t) - A_k .* S_t;                   % K x 1

    % ================================================================
    % Step 3: CVX — maximise sum of linear surrogates
    %   max   sum_k [ A_k * (gamma_k' * x(:,k)) ]    (C_k are constants)
    %   s.t.  sum_k x_mk^2 <= P_max   for each m
    %         x_mk >= 0
    % ================================================================
    cvx_quiet true
    cvx_begin
        variable x(M, K)
        obj = 0;
        for k = 1:K
            obj = obj + A_k(k) * (Gammaan(:,k)' * x(:,k));
        end
        maximize(obj)
        subject to
            x >= 0;
            for m = 1:M
                x(m,:) * x(m,:)' <= P_max;
            end
    cvx_end

    if ~contains(cvx_status, 'Solved')
        % fprintf('  KKT SCA: CVX %s at iter %d\n', cvx_status, iter);
        if iter == 1; return; end
        break;
    end

    x_cur = max(x, 0);
    R_new = dl_rate_calculate(x_cur.^2, Gammaan, BETAAn, PhiPhi);
    % fprintf('  KKT iter %2d: R_sum = %.4f  (delta = %.2e)\n', ...
    %         iter, R_new, abs(R_new - R_old));

    if abs(R_new - R_old) < tol; break; end
    R_old = R_new;
end

rho_opt = x_cur.^2;
end