function [betas, gammas, phiis, R_equal, R_frac, R_log, power_eq, power_frac, power_log, ...
          rcs_values, ap_locations, sr_locations] = downlink_ISAC_sumrate_data(...
          num_sam, num_ap, num_ue, num_sr, num_antenna, tau, power_f, ...
          Hb, Hm, f, d0, d1, D, nu)

    %% =====================================================================
    % DOWNLINK ISAC SUMRATE DATA GENERATION
    % Integrated Sensing and Communication for Cell-Free Massive MIMO
    % =====================================================================
    % Generates:
    %   - Communication data (betas, gammas, phiis, rates, power allocation)
    %   - Sensing data (RCS, target locations, CRLB values)
    % =====================================================================

    %% Initial parameters
    M = num_ap;     % number of access points
    K = num_ue;     % number of user equipments
    T = num_sr;     % number of sensing receivers
    
    % Pilot sequences
    [U, S, V] = svd(randn(tau, tau)); % U includes tau orthogonal sequences

    B = 20; % Bandwidth [MHz]
    
    % Path loss model parameters
    aL = (1.1*log10(f) - 0.7)*Hm - (1.56*log10(f) - 0.8);
    L = 46.3 + 33.9*log10(f) - 13.82*log10(Hb) - aL;

    % Power and noise parameters
    noise_p = 10^((-203.975 + 10*log10(20*10^6) + 9)/10); % noise power
    Pd = power_f / noise_p;  % normalized receive SNR
    Pp = Pd;  % pilot power

    N = num_sam;
    
    %% Initialize output arrays
    % Communication outputs
    R_cf_min = zeros(1, N);
    R_cf_user = zeros(N, K);
    R_equal = zeros(1, N);
    R_frac = zeros(1, N);
    R_log = zeros(1, N);

    betas = zeros(N, M, K);
    gammas = zeros(N, M, K);
    phiis = zeros(N, tau, K);
    power_eq = zeros(N, M, K);
    power_frac = zeros(N, M, K);
    power_log = zeros(N, M, K);
    
    % Sensing outputs
    sensing_data = struct('ap_locations', [], 'sr_locations', [], ...
                         'target_location', [], 'distances_ap_target', [], ...
                         'distances_sr_target', []);
    ap_locations = zeros(N,M,2);
    sr_locations = zeros(N,T,2);
    rcs_values = zeros(N, M, T);  % RCS values for each AP-SR pair
    crlb_values = zeros(N, 2);     % CRLB for x and y localization
    
    % Sensing system parameters
    c = 3e8;  % speed of light [m/s]
    sigma_s = 1e-3;  % noise power at sensing receivers (normalized)
    B_sens = B * 1e6;  % bandwidth in Hz
    zeta_coeff = 8 * pi^2 * B_sens^2 / (sigma_s^2 * c^2);

    fprintf('====== %d samples, %d UEs, %d APs, %d SRs ... at %s \n', ...
            num_sam, num_ue, num_ap, num_sr, char(datetime('now')));
    
    for n = 1:N
        if mod(n, max(1, N/50)) == 1
            fprintf('============== %d/%d ==============  at %s \n', ...
                    n, N, char(datetime('now')));
        end
        
        %% =========== LOCATIONS AND GEOMETRY ===========
        
        % Access Point locations (with wraparound 9-cell layout)
        AP = zeros(M, 2, 9);
        AP(:,:,1) = unifrnd(-D/2, D/2, M, 2);
        
        % Wraparound neighbors (8 neighbor cells)
        D1 = zeros(M, 2); D1(:,1) = D1(:,1) + D*ones(M,1);
        AP(:,:,2) = AP(:,:,1) + D1;
        D2 = zeros(M, 2); D2(:,2) = D2(:,2) + D*ones(M,1);
        AP(:,:,3) = AP(:,:,1) + D2;
        D3 = zeros(M, 2); D3(:,1) = D3(:,1) - D*ones(M,1);
        AP(:,:,4) = AP(:,:,1) + D3;
        D4 = zeros(M, 2); D4(:,2) = D4(:,2) - D*ones(M,1);
        AP(:,:,5) = AP(:,:,1) + D4;
        D5 = zeros(M, 2); D5(:,1) = D5(:,1) + D*ones(M,1); D5(:,2) = D5(:,2) - D*ones(M,1);
        AP(:,:,6) = AP(:,:,1) + D5;
        D6 = zeros(M, 2); D6(:,1) = D6(:,1) - D*ones(M,1); D6(:,2) = D6(:,2) + D*ones(M,1);
        AP(:,:,7) = AP(:,:,1) + D6;
        D7 = zeros(M, 2); D7 = D7 + D*ones(M,2);
        AP(:,:,8) = AP(:,:,1) + D7;
        D8 = zeros(M, 2); D8 = D8 - D*ones(M,2);
        AP(:,:,9) = AP(:,:,1) + D8;
        
        % User Equipment locations (with wraparound)
        Ter = zeros(K, 2, 9);
        Ter(:,:,1) = unifrnd(-D/2, D/2, K, 2);
        
        D1 = zeros(K, 2); D1(:,1) = D1(:,1) + D*ones(K,1);
        Ter(:,:,2) = Ter(:,:,1) + D1;
        D2 = zeros(K, 2); D2(:,2) = D2(:,2) + D*ones(K,1);
        Ter(:,:,3) = Ter(:,:,1) + D2;
        D3 = zeros(K, 2); D3(:,1) = D3(:,1) - D*ones(K,1);
        Ter(:,:,4) = Ter(:,:,1) + D3;
        D4 = zeros(K, 2); D4(:,2) = D4(:,2) - D*ones(K,1);
        Ter(:,:,5) = Ter(:,:,1) + D4;
        D5 = zeros(K, 2); D5(:,1) = D5(:,1) + D*ones(K,1); D5(:,2) = D5(:,2) - D*ones(K,1);
        Ter(:,:,6) = Ter(:,:,1) + D5;
        D6 = zeros(K, 2); D6(:,1) = D6(:,1) - D*ones(K,1); D6(:,2) = D6(:,2) + D*ones(K,1);
        Ter(:,:,7) = Ter(:,:,1) + D6;
        D7 = zeros(K, 2); D7 = D7 + D*ones(K,2);
        Ter(:,:,8) = Ter(:,:,1) + D7;
        D8 = zeros(K, 2); D8 = D8 - D*ones(K,2);
        Ter(:,:,9) = Ter(:,:,1) + D8;
        
        % Sensing Receiver locations (distributed, similar to APs)
        SR = zeros(T, 2, 9);
        SR(:,:,1) = unifrnd(-D/2, D/2, T, 2);
        
        D1 = zeros(T, 2); D1(:,1) = D1(:,1) + D*ones(T,1);
        SR(:,:,2) = SR(:,:,1) + D1;
        D2 = zeros(T, 2); D2(:,2) = D2(:,2) + D*ones(T,1);
        SR(:,:,3) = SR(:,:,1) + D2;
        D3 = zeros(T, 2); D3(:,1) = D3(:,1) - D*ones(T,1);
        SR(:,:,4) = SR(:,:,1) + D3;
        D4 = zeros(T, 2); D4(:,2) = D4(:,2) - D*ones(T,1);
        SR(:,:,5) = SR(:,:,1) + D4;
        D5 = zeros(T, 2); D5(:,1) = D5(:,1) + D*ones(T,1); D5(:,2) = D5(:,2) - D*ones(T,1);
        SR(:,:,6) = SR(:,:,1) + D5;
        D6 = zeros(T, 2); D6(:,1) = D6(:,1) - D*ones(T,1); D6(:,2) = D6(:,2) + D*ones(T,1);
        SR(:,:,7) = SR(:,:,1) + D6;
        D7 = zeros(T, 2); D7 = D7 + D*ones(T,2);
        SR(:,:,8) = SR(:,:,1) + D7;
        D8 = zeros(T, 2); D8 = D8 - D*ones(T,2);
        SR(:,:,9) = SR(:,:,1) + D8;
        
        % Target location (center of cell for simplicity)
        target_loc = [0, 0];
        
        %% =========== COMMUNICATION: SHADOWING AND LARGE-SCALE FADING ===========
        
        sigma_shd = 8;  % shadowing std dev [dB]
        D_cor = 0.1;    % correlation distance
        
        % AP shadowing coefficients
        Dist_AP = zeros(M, M);
        Cor_AP = zeros(M, M);
        for m1 = 1:M
            for m2 = 1:M
                min_dist = min([norm(AP(m1,:,1)-AP(m2,:,1)), norm(AP(m1,:,1)-AP(m2,:,2)), ...
                    norm(AP(m1,:,1)-AP(m2,:,3)), norm(AP(m1,:,1)-AP(m2,:,4)), ...
                    norm(AP(m1,:,1)-AP(m2,:,5)), norm(AP(m1,:,1)-AP(m2,:,6)), ...
                    norm(AP(m1,:,1)-AP(m2,:,7)), norm(AP(m1,:,1)-AP(m2,:,8)), ...
                    norm(AP(m1,:,1)-AP(m2,:,9))]);
                Dist_AP(m1,m2) = min_dist;
                Cor_AP(m1,m2) = exp(-log(2)*min_dist/D_cor);
            end
        end
        A1 = chol(Cor_AP,'lower');
        x1 = randn(M,1);
        sh_AP = A1*x1;
        for m = 1:M
            sh_AP(m) = (1/sqrt(2))*sigma_shd*sh_AP(m)/norm(A1(m,:));
        end
        
        % UE shadowing coefficients
        Dist_UE = zeros(K, K);
        Cor_UE = zeros(K, K);
        for k1 = 1:K
            for k2 = 1:K
                min_dist = min([norm(Ter(k1,:,1)-Ter(k2,:,1)), norm(Ter(k1,:,1)-Ter(k2,:,2)), ...
                    norm(Ter(k1,:,1)-Ter(k2,:,3)), norm(Ter(k1,:,1)-Ter(k2,:,4)), ...
                    norm(Ter(k1,:,1)-Ter(k2,:,5)), norm(Ter(k1,:,1)-Ter(k2,:,6)), ...
                    norm(Ter(k1,:,1)-Ter(k2,:,7)), norm(Ter(k1,:,1)-Ter(k2,:,8)), ...
                    norm(Ter(k1,:,1)-Ter(k2,:,9))]);
                Dist_UE(k1,k2) = min_dist;
                Cor_UE(k1,k2) = exp(-log(2)*min_dist/D_cor);
            end
        end
        A2 = chol(Cor_UE,'lower');
        x2 = randn(K,1);
        sh_Ter = A2*x2;
        for k = 1:K
            sh_Ter(k) = (1/sqrt(2))*sigma_shd*sh_Ter(k)/norm(A2(k,:));
        end
        
        % Shadowing matrix
        Z_shd = zeros(M,K);
        for m = 1:M
            for k = 1:K
                Z_shd(m,k) = sh_AP(m) + sh_Ter(k);
            end
        end
        
        %% =========== COMMUNICATION: LARGE-SCALE FADING COEFFICIENTS ===========
        
        BETAA = zeros(M, K);
        dist_mk = zeros(M, K);
        
        for m = 1:M
            for k = 1:K
                % Find minimum distance over all wraparound cells
                min_dist = norm(AP(m,:,1) - Ter(k,:,1));
                for cell = 2:9
                    dist_temp = norm(AP(m,:,cell) - Ter(k,:,1));
                    if dist_temp < min_dist
                        min_dist = dist_temp;
                    end
                end
                dist_mk(m,k) = min_dist;
                
                % Path loss calculation
                if dist_mk(m,k) < d0
                    betadB = -L - 35*log10(d1) + 20*log10(d1) - 20*log10(d0);
                elseif (dist_mk(m,k) >= d0) && (dist_mk(m,k) <= d1)
                    betadB = -L - 35*log10(d1) + 20*log10(d1) - 20*log10(dist_mk(m,k));
                else
                    betadB = -L - 35*log10(dist_mk(m,k)) + Z_shd(m,k);
                end
                
                BETAA(m,k) = 10^(betadB/10);
            end
        end
        
        %% =========== PILOT ASSIGNMENT ===========
        
        Phii = zeros(tau, K);
        for k = 1:K
            Point = randi([1, tau]);
            Phii(:,k) = U(:,Point);
        end
        
        %% =========== CHANNEL ESTIMATION ===========
        
        Phii_cf = Phii;  % pilot set for cell-free
        
        % Gamma matrix (channel estimate variances)
        Gammaa = zeros(M, K);
        mau = zeros(M, K);
        for m = 1:M
            for k = 1:K
                mau(m,k) = norm((BETAA(m,:).^(1/2)).*(Phii_cf(:,k)'*Phii_cf))^2;
            end
        end
        
        for m = 1:M
            for k = 1:K
                Gammaa(m,k) = tau*Pp*BETAA(m,k)^2 / (tau*Pp*mau(m,k) + 1);
            end
        end
        
        %% =========== GREEDY PILOT ASSIGNMENT ===========
        
        stepp = 5;
        Ratestep = zeros(stepp, K);
        
        % Compute SINR and rates with initial pilot assignment
        etaa = zeros(M, 1);
        for m = 1:M
            etaa(m) = 1 / sum(Gammaa(m,:));
        end
        
        SINR = zeros(1, K);
        for k = 1:K
            num = 0;
            for m = 1:M
                num = num + (etaa(m)^(1/2))*Gammaa(m,k);
            end
            SINR(k) = (num_antenna^2)*Pd*num^2 / (1 + (num_antenna)*Pd*sum(BETAA(:,k)));
            Ratestep(1,k) = log2(1 + SINR(k));
        end
        
        % Greedy refinement
        for st = 2:stepp
            [~, minindex] = min(Ratestep(st-1,:));
            
            Mat = zeros(tau, tau) - Pd*sum(BETAA(:,minindex))*Phii_cf(:,minindex)*Phii_cf(:,minindex)';
            for mm = 1:M
                for kk = 1:K
                    Mat = Mat + Pd*BETAA(mm,kk)*Phii_cf(:,kk)*Phii_cf(:,kk)';
                end
            end
            [U1, ~, ~] = svd(Mat);
            Phii_cf(:,minindex) = U1(:,tau);
            
            % Update gamma matrix
            Gammaa = zeros(M, K);
            mau = zeros(M, K);
            for m = 1:M
                for k = 1:K
                    mau(m,k) = norm((BETAA(m,:).^(1/2)).*(Phii_cf(:,k)'*Phii_cf))^2;
                end
            end
            
            for m = 1:M
                for k = 1:K
                    Gammaa(m,k) = tau*Pp*BETAA(m,k)^2 / (tau*Pp*mau(m,k) + 1);
                end
            end
            
            % Recompute rates
            etaa = zeros(M, 1);
            for m = 1:M
                etaa(m) = 1 / sum(Gammaa(m,:));
            end
            
            for k = 1:K
                num = 0;
                for m = 1:M
                    num = num + (etaa(m)^(1/2))*Gammaa(m,k);
                end
                SINR(k) = (num_antenna^2)*Pd*num^2 / (1 + (num_antenna)*Pd*sum(BETAA(:,k)));
                Ratestep(st,k) = log2(1 + SINR(k));
            end
        end
        
        R_cf_min(n) = min(Ratestep(stepp,:));
        R_cf_user(n,:) = Ratestep(stepp,:);
        
        %% =========== PRE-COMPUTE FOR POWER ALLOCATION ===========
        
        PhiPhi = zeros(K, K);
        for ii = 1:K
            for k = 1:K
                PhiPhi(ii,k) = norm(Phii_cf(:,ii)'*Phii_cf(:,k));
            end
        end
        BETAAn = BETAA*Pd;
        Gammaan = Gammaa*Pd;
        
        %% =========== SENSING CALCULATIONS ===========
        
        % Compute distances from APs to target
        dist_ap_target = zeros(M, 1);
        for m = 1:M
            dist_ap_target(m) = norm(AP(m,:,1) - target_loc);
        end
        
        % Compute distances from SRs to target
        dist_sr_target = zeros(T, 1);
        for t = 1:T
            dist_sr_target(t) = norm(SR(t,:,1) - target_loc);
        end
        
        % Radar Cross Section (RCS) for each AP-SR pair
        % RCS reflects the radar reflectivity of the target
        % Model: chi_mt ~ CN(0, RCS_mt) where RCS depends on geometry
        rcs_mt = zeros(M, T);
        for m = 1:M
            for t = 1:T
                % Simple RCS model: distance-dependent
                % RCS decreases with combined AP-SR distance
                combined_dist = dist_ap_target(m) + dist_sr_target(t);
                % Normalized RCS with some randomness
                rcs_mt(m,t) = (1e-3 / (combined_dist + eps))^2 * abs(randn()).^2;
            end
        end
        
        % CRLB computation for target localization
        % Following equation (5) from the paper
        % We compute the inverse of the Fisher Information Matrix
        
        % Prepare sensing geometry matrices
        q_a = zeros(M, 1);  % x-component geometry
        q_b = zeros(M, 1);  % y-component geometry
        q_c = zeros(M, 1);  % cross-component geometry
        
        for m = 1:M
            sum_x_term = 0;
            sum_y_term = 0;
            sum_xy_term = 0;
            
            for t = 1:T
                % Geometric terms for Cramer-Rao bound
                x_ap = AP(m,1,1);
                y_ap = AP(m,2,1);
                x_sr = SR(t,1,1);
                y_sr = SR(t,2,1);
                x_tgt = target_loc(1);
                y_tgt = target_loc(2);
                
                % Direction cosines
                dc_ap_x = (x_ap - x_tgt) / (dist_ap_target(m) + eps);
                dc_ap_y = (y_ap - y_tgt) / (dist_ap_target(m) + eps);
                dc_sr_x = (x_sr - x_tgt) / (dist_sr_target(t) + eps);
                dc_sr_y = (y_sr - y_tgt) / (dist_sr_target(t) + eps);
                
                % Gradient terms
                grad_x = dc_ap_x + dc_sr_x;
                grad_y = dc_ap_y + dc_sr_y;
                
                % Accumulate Fisher Information
                sum_x_term = sum_x_term + rcs_mt(m,t) * grad_x^2;
                sum_y_term = sum_y_term + rcs_mt(m,t) * grad_y^2;
                sum_xy_term = sum_xy_term + rcs_mt(m,t) * grad_x * grad_y;
            end
            
            q_a(m) = zeta_coeff * sum_x_term;
            q_b(m) = zeta_coeff * sum_y_term;
            q_c(m) = zeta_coeff * sum_xy_term;
        end
        
        % % Fisher Information Matrix
        % FIM = [sum(q_a), sum(q_c); sum(q_c), sum(q_b)];
        % 
        % % CRLB is the inverse of Fisher Information Matrix
        % if det(FIM) > 1e-10
        %     CRLB_matrix = inv(FIM);
        %     crlb_x = CRLB_matrix(1,1);
        %     crlb_y = CRLB_matrix(2,2);
        % else
        %     crlb_x = Inf;
        %     crlb_y = Inf;
        % end

        b = q_a + q_b;
        A = q_a * q_b' + q_c * q_c';

        %% =========== POWER ALLOCATION METHODS ===========
        
        % 1) Equal Power Allocation (baseline)
        P_max = 1;
        rho_eq = (P_max / K) * ones(M, K);
        kappa_eq = crlb_linear_check(rho_eq, b, A, nu);
        rho_eq = rho_eq * kappa_eq;
        R_dl_equal = dl_rate_calculate(rho_eq, Gammaan, BETAAn, PhiPhi);

        
        % 2) Fractional Power Allocation
        theta = 1.0;
        rho_frac = dl_fractional_pa(BETAA, M, K, P_max, theta);
        kappa_frac = crlb_linear_check(rho_frac, b, A, nu);
        rho_frac = rho_frac * kappa_frac;
        R_dl_frac = dl_rate_calculate(rho_frac, Gammaan, BETAAn, PhiPhi);
        
        % 3) Logarithmic Approximation (simplified)
        [R_dl_log, rho_log] = dl_approx_sumrate(Gammaan, BETAAn, PhiPhi, P_max);
        
        
        %% =========== SAVE RESULTS ===========
        
        % Communication results
        betas(n,:,:) = BETAAn;
        gammas(n,:,:) = Gammaan;
        phiis(n,:,:) = Phii_cf;
        R_equal(n) = R_dl_equal;
        R_frac(n) = R_dl_frac;
        R_log(n) = R_dl_log;
        power_eq(n,:,:) = rho_eq;
        power_frac(n,:,:) = rho_frac;
        power_log(n,:,:) = rho_log;
        
        % Sensing results
        rcs_values(n,:,:) = rcs_mt;
        % crlb_values(n,1) = crlb_x;
        % crlb_values(n,2) = crlb_y;
        
        % % Store location data
        % if n == 1
        %     sensing_data.ap_locations = AP(:,:,1);
        %     sensing_data.sr_locations = SR(:,:,1);
        % end
        % sensing_data.target_location(n,:) = target_loc;
        % sensing_data.distances_ap_target(n,:) = dist_ap_target';
        % sensing_data.distances_sr_target(n,:) = dist_sr_target';

        ap_locations(n,:,:) = AP(:,:,1);
        sr_locations(n,:,:) = SR(:,:,1);
        
    end  % end sample loop

end  % end main function