% 1. Create dummy GPU matrices
M_test = 500; K_test = 100;
rho_g     = gpuArray(rand(M_test, K_test));
Gammaan_g = gpuArray(rand(M_test, K_test));
BETAAn_g  = gpuArray(rand(M_test, K_test));
PhiPhi_g  = gpuArray(rand(K_test, K_test));

% 2. Run a heavy loop to keep the GPU busy
tic;
for i = 1:50000
    % Using the optimized function
    sinr_g = dl_sinr_calculate(rho_g, Gammaan_g, BETAAn_g, PhiPhi_g);
end
toc;