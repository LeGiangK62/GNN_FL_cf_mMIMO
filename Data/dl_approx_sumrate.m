function [rate, rho_opt] = dl_approx_sumrate(Gammaan, BETAAn, PhiPhi, P_max)
    max_iter = 50;
    tol      = 1e-4;

    [M,K] = size(Gammaan);
    cur_rho = (P_max / K) * ones(M, K);
    cur_rate = dl_rate_calculate(cur_rho, Gammaan, BETAAn, PhiPhi);
    
    for iter = 1:max_iter
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
        
            sig = sum(x .* Gammaan, 1);
            PC  = sum(BETAAn' * (square(x) .* Gammaan), 2)';
        
            V        = x .* Gammaan ./ BETAAn;
            inner    = BETAAn' * V;
            inner_sq = square(inner);
            UI       = sum((inner_sq .* (1-eye(K))) .* PhiPhi, 2)';
        
            denom_expr = 1 + PC + UI;
        
            obj = sum(alpha .* (2*log(sig)/log(2) - denom_expr./(log(2)*denom)));
        
            maximize(obj)
            subject to
                x >= 0
                sum_square(x,2) <= P_max
        cvx_end

        if ~contains(cvx_status, 'Solved')
            % fprintf('  Log-Approx: CVX %s at iter %d\n', cvx_status, iter);
            if iter == 1
                % new_rho = cur_rho; 
                rho_opt = cur_rho;
                rate    = cur_rate;
                return; 
            end
            break;
        end
        new_rho = x.^2;
        new_rate = dl_rate_calculate(new_rho, Gammaan, BETAAn, PhiPhi);
        if abs(new_rate - cur_rate) / max(abs(cur_rate),1) < tol 
            break; 
        end
        cur_rate = new_rate;
        cur_rho = new_rho;
        % eta = 0.5;   % hoặc 0.7, 0.8
        % cur_rho = (1-eta)*cur_rho + eta*new_rho;
        % fprintf('Log-Approx: %f at iter %d\n', cur_rate, iter);

    end
    rho_opt = cur_rho;
    rate = cur_rate; % Assign the final computed rate

    
        
end
