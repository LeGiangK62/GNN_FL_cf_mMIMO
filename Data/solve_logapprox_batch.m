function [power_log, R_log, status, converged] = solve_logapprox_batch( ...
        betas, Gammas, Phii_cf, q_a_all, q_b_all, q_c_all, ...
        nu, power_f, num_antenna, max_iter)
%SOLVE_LOGAPPROX_BATCH Solve independent weighted-power ISAC instances.
%   betas and Gammas are already normalized by power_f/noise power in the
%   dataset generator. power_f is retained in this public signature so the
%   wrapper matches the generator configuration without rescaling twice.

    arguments
        betas double
        Gammas double
        Phii_cf double
        q_a_all double
        q_b_all double
        q_c_all double
        nu (1,1) double {mustBePositive}
        power_f (1,1) double {mustBePositive} %#ok<INUSD>
        num_antenna (1,1) double {mustBePositive, mustBeInteger}
        max_iter (1,1) double {mustBePositive, mustBeInteger} = 50
    end

    [num_sam, num_ap, num_ue] = size(Gammas);
    assert(isequal(size(betas), [num_sam, num_ap, num_ue]), ...
        'betas and Gammas must have identical [N,M,K] dimensions.');
    assert(size(Phii_cf, 1) == num_sam && size(Phii_cf, 3) == num_ue, ...
        'Phii_cf must have dimensions [N,tau,K].');

    power_log = zeros(num_sam, num_ap, num_ue);
    R_log = zeros(1, num_sam);
    status = zeros(num_sam, 1, 'uint8');
    converged = false(num_sam, 1);
    P_max = 1 / num_antenna;

    requested_workers = 12;
    pool = gcp('nocreate');
    if ~isempty(pool) && (~isa(pool, 'parallel.ProcessPool') || ...
            pool.NumWorkers ~= requested_workers)
        delete(pool);
        pool = [];
    end
    if isempty(pool)
        pool = parpool('Processes', requested_workers);
    end
    fprintf('solve_logapprox_batch: using %d parfor workers.\n', ...
        pool.NumWorkers);
    persistent successive_warning_has_been_shown
    show_successive_warning = isempty(successive_warning_has_been_shown);
    successive_warning_has_been_shown = true;
    parfor n = 1:num_sam
        [power_log(n,:,:), R_log(n), status(n), converged(n)] = solve_one( ...
            betas, Gammas, Phii_cf, q_a_all, q_b_all, q_c_all, ...
            n, num_ap, num_ue, P_max, nu, max_iter, ...
            show_successive_warning && n == 1);
    end
end


function [power_n, rate_n, status_n, converged_n] = solve_one( ...
        betas, Gammas, Phii_cf, q_a_all, q_b_all, q_c_all, ...
        n, num_ap, num_ue, P_max, nu, max_iter, show_successive_warning)
    % CVX prints "NOTE: custom settings have been set for this solver."
    % whenever cvx_solver_settings is called.  cvx_quiet does not suppress
    % that notice, so capture this small per-realization setup block.
    setup_output = evalc([ ...
        'maxNumCompThreads(1); ' ...
        'cvx_solver mosek; ' ...
        'cvx_solver_settings(''MSK_IPAR_NUM_THREADS'', 1); ' ...
        'cvx_quiet(true);']); %#ok<NASGU>
    % CVX documents cvx_expert(true) as suppressing its repeated warning
    % about the experimental successive-approximation method. Leave it off
    % for exactly the first realization of the first wrapper invocation.
    cvx_expert(~show_successive_warning);

    beta_n = reshape(betas(n,:,:), num_ap, num_ue);
    gamma_n = reshape(Gammas(n,:,:), num_ap, num_ue);
    phi_n = reshape(Phii_cf(n,:,:), size(Phii_cf, 2), num_ue);
    PhiPhi = abs(phi_n' * phi_n);
    q_a = reshape(q_a_all(n,:,:), num_ap, 1);
    q_b = reshape(q_b_all(n,:,:), num_ap, 1);
    q_c = reshape(q_c_all(n,:,:), num_ap, 1);

    if show_successive_warning
        [rate_n, power_n, raw_status, ~, ~, converged_n] = ...
            dl_isac_approx_sumrate(gamma_n, beta_n, PhiPhi, P_max, ...
            q_a, q_b, q_c, nu, max_iter);
    else
        solver_output = evalc( ...
            ['[rate_n, power_n, raw_status, ~, ~, converged_n] = ' ...
             'dl_isac_approx_sumrate(gamma_n, beta_n, PhiPhi, P_max, ' ...
             'q_a, q_b, q_c, nu, max_iter);']); %#ok<NASGU>
    end
    status_n = uint8(raw_status);
    if status_n == 2
        % Preserve captured CVX/MOSEK diagnostics for genuine failures.
        if ~isempty(strtrim(setup_output)), fprintf(2, '%s', setup_output); end
        if exist('solver_output', 'var') && ~isempty(strtrim(solver_output))
            fprintf(2, '%s', solver_output);
        end
    end
end
