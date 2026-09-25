function [rate, rho_opt, status, iteration_count, trace, converged] = ...
        dl_isac_approx_sumrate( ...
        Gammaan, BETAAn, PhiPhi, P_max, q_a, q_b, q_c, nu, max_iter)
    % Successive convex approximation for the sum-rate problem with the
    % scalar localization-CRLB constraint
    %
    %   p_m = sum_k Gammaan(m,k)*rho(m,k)
    %   (q_a+q_b)'p - nu*((q_a'p)*(q_b'p) - (q_c'p)^2) <= 0.
    %
    % CVX cannot use p=sum_square(x,2) inside the indefinite quadratic
    % expression.  At every iteration we therefore linearize both p=x.^2
    % and the FIM determinant about the previous feasible point.  The CVX
    % subproblem then contains an affine CRLB constraint.  A backtracking
    % step checks the TRUE factored constraint before accepting an update.
    if nargin < 9 || isempty(max_iter)
        max_iter = 50;
    end
    tol      = 1e-4;
    trust_radius = 0.35 * sqrt(P_max);
    fim_eps = 1e-12;

    [M,K] = size(Gammaan);
    status = 0;
    iteration_count = 0;
    converged = false;
    trace = struct( ...
        'rate', nan(max_iter, 1), ...
        'max_constraint_violation', nan(max_iter, 1), ...
        'rho_relative_change', nan(max_iter, 1), ...
        'cvx_wall_time', nan(max_iter, 1), ...
        'cvx_cpu_time', nan(max_iter, 1), ...
        'cvx_solver_iterations', nan(max_iter, 1));
    cur_rho = P_max ./ (K * Gammaan);
    cur_x = sqrt(cur_rho);
    cur_rate = dl_rate_calculate(cur_rho, Gammaan, BETAAn, PhiPhi);

    p_t = sum(cur_rho .* Gammaan, 2);
    [cur_sigma2, cur_feasible] = local_sigma2(p_t, q_a, q_b, q_c, nu, fim_eps);
    if ~cur_feasible
        % Find a feasible sensing-power vector with an exact 2x2 Schur-
        % complement SDP: W >= inv(F), trace(W) <= nu.  This initialization
        % is convex because F is affine in rho through
        % p_m=sum_k Gammaan(m,k)*rho(m,k).
        [p_init, init_status] = local_feasible_initial_power( ...
            q_a, q_b, q_c, P_max, nu);
        if init_status ~= 0
            if init_status == 1
                warning('dl_isac_approx_sumrate:InfeasibleProblem', ...
                    ['No feasible CRLB initialization was found ' ...
                     '(equal-power sigma2=%g, nu=%g).'], cur_sigma2, nu);
            else
                warning('dl_isac_approx_sumrate:InitializerFailure', ...
                    ['The feasible initializer failed ' ...
                     '(equal-power sigma2=%g, nu=%g).'], cur_sigma2, nu);
            end
            rho_opt = cur_rho;
            rate = cur_rate;
            status = init_status;
            trace = trim_trace(trace, iteration_count);
            return;
        end
        cur_rho = p_init ./ (K * Gammaan);
        cur_x = sqrt(cur_rho);
        cur_rate = dl_rate_calculate(cur_rho, Gammaan, BETAAn, PhiPhi);
        p_t = sum(cur_rho .* Gammaan, 2);
    end
    
    for iter = 1:max_iter
        iteration_count = iter;
        % fprintf('\t iter %d/%d ======= \n', iter, max_iter);
        
        [alpha, ~, denom] = log_approximation(cur_rho, Gammaan, BETAAn, PhiPhi);
           
        %%
        % cvx_quiet true
        % cvx_begin
        %     variable x(M, K)
        %     obj = 0;
        %     for k = 1:K
        %         % num term
        %         obj = obj + alpha(k) * 2 * log(x(:,k)' * Gammaan(:,k))/ log(2);
        %         % denom term
        %         PC = 0; UI = 0;
        %         for k_prime = 1:K
        %             for m = 1:M
        %                 PC = PC + (x(m,k_prime)^2)*Gammaan(m,k_prime)*BETAAn(m,k);
        %             end
        % 
        %             if k ~= k_prime
        %                 tmp = 0;
        %                 for m = 1:M
        %                     tmp = tmp + x(m,k_prime) * Gammaan(m,k_prime) * BETAAn(m,k) / BETAAn(m,k_prime);
        %                 end
        %                 UI = UI + tmp^2*PhiPhi(k,k_prime);
        %             end
        %         end
        %         tmp = (1 + PC + UI)/ (log(2) * denom(k));
        %         % tmp = log(1 + PC + UI)/log(2);
        %         obj = obj - alpha(k) * tmp;
        %     end
        % 
        %     maximize(obj)
        %     subject to
        %         x >= 0;
        %         for m = 1:M
        %             x(m,:) * x(m,:)' <= P_max;
        %         end
        % cvx_end
        %% Vectorize
        
        cvx_timer = tic;
        cvx_begin quiet
        cvx_solver mosek
            variable x(M,K)
        
            expression sig(1,K)
            expression PC(1,K)
            expression UI(1,K)
            expression denom_expr(1,K)
            expression V(M,K)
            expression inner(K,K)
            expression inner_sq(K,K)
            expression obj
            expression p_lin(M)
            expression det_lin
            expression crlb_lin
        
            sig = sum(x .* Gammaan, 1);
            PC  = sum(BETAAn' * (square(x) .* Gammaan), 2)';
        
            V        = x .* Gammaan ./ BETAAn;
            inner    = BETAAn' * V;
            inner_sq = square(inner);
            UI       = sum((inner_sq .* (1-eye(K))) .* PhiPhi, 2)';
        
            denom_expr = 1 + PC + UI;

            % First-order model of
            % p_m=sum_k Gammaan(m,k)*x_mk^2 at cur_x.
            % This is affine in the current CVX variable x.
            p_lin = 2 * sum(Gammaan .* cur_x .* x, 2) ...
                    - sum(Gammaan .* cur_x.^2, 2);

            % First-order model of det(FIM) at p_t, written in factored
            % scalar form.  Do not construct A.
            sa_t = q_a' * p_t;
            sb_t = q_b' * p_t;
            sc_t = q_c' * p_t;
            det_t = sa_t * sb_t - sc_t^2;
            grad_det = q_a * sb_t + q_b * sa_t - 2 * q_c * sc_t;
            det_lin = det_t + grad_det' * (p_lin - p_t);
            crlb_lin = (q_a + q_b)' * p_lin - nu * det_lin;
        
            obj = sum(alpha .* (2*log(sig)/log(2) - denom_expr./(log(2)*denom)));
        
            maximize(obj)
            subject to
                x >= 0
                p_lin >= 0
                crlb_lin <= 0
                sum(Gammaan .* square(x), 2) <= P_max
                norm(x - cur_x, 'fro') <= trust_radius

        cvx_end
        trace.cvx_wall_time(iter) = toc(cvx_timer);
        trace.cvx_cpu_time(iter) = cvx_cputime;
        trace.cvx_solver_iterations(iter) = cvx_slvitr;

        if ~contains(cvx_status, 'Solved')
            % fprintf('  Log-Approx: CVX %s at iter %d\n', cvx_status, iter);
            if iter == 1
                % new_rho = cur_rho; 
                rho_opt = cur_rho;
                rate    = cur_rate;
                status  = 2;
                trace = trim_trace(trace, iteration_count);
                return; 
            end
            status = 2;
            break;
        end
        % Backtrack from the previous feasible point until the TRUE CRLB
        % constraint (not its affine model) is satisfied.
        eta = 1.0;
        accepted = false;
        while eta >= 2^-12
            trial_x = (1-eta) * cur_x + eta * x;
            trial_rho = trial_x.^2;
            trial_p = sum(trial_rho .* Gammaan, 2);
            [~, trial_feasible] = local_sigma2( ...
                trial_p, q_a, q_b, q_c, nu, fim_eps);
            if trial_feasible
                accepted = true;
                break;
            end
            eta = eta / 2;
        end
        if ~accepted
            warning('dl_isac_approx_sumrate:BacktrackingFailed', ...
                'No feasible SCA step at iteration %d; keeping previous point.', iter);
            status = 2;
            break;
        end

        new_rho = trial_rho;
        new_rate = dl_rate_calculate(new_rho, Gammaan, BETAAn, PhiPhi);
        new_p = sum(new_rho .* Gammaan, 2);
        sa_new = q_a' * new_p;
        sb_new = q_b' * new_p;
        sc_new = q_c' * new_p;
        crlb_violation = (sa_new + sb_new) ...
            - nu * (sa_new * sb_new - sc_new^2);
        power_violation = max(new_p - P_max);
        nonnegative_violation = max(-new_rho(:));
        trace.rate(iter) = new_rate;
        trace.max_constraint_violation(iter) = max([ ...
            0, crlb_violation, power_violation, nonnegative_violation]);
        trace.rho_relative_change(iter) = norm( ...
            new_rho - cur_rho, 'fro') / max(norm(cur_rho, 'fro'), eps);
        if abs(new_rate - cur_rate) / max(abs(cur_rate),1) < tol 
            cur_rate = new_rate;
            cur_rho = new_rho;
            converged = true;
            break; 
        end
        cur_rate = new_rate;
        cur_rho = new_rho;
        cur_x = sqrt(cur_rho);
        p_t = sum(cur_rho .* Gammaan, 2);
        % fprintf('Log-Approx: %f at iter %d\n', cur_rate, iter);

    end
    rho_opt = cur_rho;
    rate = cur_rate; % Assign the final computed rate

    trace = trim_trace(trace, iteration_count);

    
        
end


function trace = trim_trace(trace, iteration_count)
    trace_fields = fieldnames(trace);
    for field_idx = 1:numel(trace_fields)
        field_name = trace_fields{field_idx};
        trace.(field_name) = trace.(field_name)(1:iteration_count);
    end
end


function [sigma2, feasible] = local_sigma2(p, q_a, q_b, q_c, nu, eps_det)
    sa = q_a' * p;
    sb = q_b' * p;
    sc = q_c' * p;
    denom = sa * sb - sc^2;
    if denom <= eps_det
        sigma2 = Inf;
        feasible = false;
        return;
    end
    sigma2 = ((q_a + q_b)' * p) / denom;
    feasible = isfinite(sigma2) && sigma2 <= nu * (1 + 1e-8);
end


function [p_init, status] = local_feasible_initial_power( ...
        q_a, q_b, q_c, P_max, nu)
    M = length(q_a);
    cvx_begin quiet
        cvx_solver mosek
        variable p0(M)
        variable W(2,2) symmetric
        expression Fim(2,2)
        Fim = [q_a' * p0, q_c' * p0; ...
               q_c' * p0, q_b' * p0];
        maximize(sum(p0))
        subject to
            0 <= p0 <= P_max
            [Fim, eye(2); eye(2), W] == semidefinite(4)
            trace(W) <= nu
    cvx_end

    if contains(cvx_status, 'Solved')
        p_init = max(0, min(P_max, p0));
        status = 0;
    elseif contains(cvx_status, 'Infeasible')
        p_init = [];
        status = 1;
    else
        p_init = [];
        status = 2;
    end
end
