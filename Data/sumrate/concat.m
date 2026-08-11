num_AP = 80;

% Tạo tên file tự động dựa vào num_AP (nếu tên file thay đổi theo số AP)
% Hoặc bạn có thể giữ nguyên chuỗi tên file cứng nếu tên file của bạn luôn cố định là '500'
file1 = sprintf('01_dl_sumrate_data_500_20_%d_20.mat', num_AP);
file2 = sprintf('02_dl_sumrate_data_500_20_%d_20.mat', num_AP);

data1 = load(file1, 'betas', 'Gammas', 'Phii_cf', 'R_equal', 'R_frac', 'R_log', 'power');
data2 = load(file2, 'betas', 'Gammas', 'Phii_cf', 'R_equal', 'R_frac', 'R_log', 'power');

betas   = [data1.betas;   data2.betas];
Gammas  = [data1.Gammas;  data2.Gammas];
Phii_cf = [data1.Phii_cf; data2.Phii_cf];
R_equal = [data1.R_equal, data2.R_equal];
R_frac  = [data1.R_frac,  data2.R_frac];
R_log   = [data1.R_log,   data2.R_log];
power   = [data1.power;   data2.power];

% 3. Đưa vào một struct tạm để duyệt và in kích thước tự động
vars = struct('betas', betas, 'Gammas', Gammas, 'Phii_cf', Phii_cf, ...
              'R_equal', R_equal, 'R_frac', R_frac, 'R_log', R_log, 'power', power);

fields = fieldnames(vars);

fprintf('\n=========================================\n');
fprintf('   KÍCH THƯỚC (SHAPE) CÁC BIẾN SAU GỘP    \n');
fprintf('=========================================\n');

for i = 1:numel(fields)
    name = fields{i};
    sz   = size(vars.(name));
    szStr = strjoin(string(sz), ' x '); % Chuyển kích thước thành dạng "500 x 20 x 40"
    
    fprintf('%-10s : [%s]\n', name, szStr);
end
fprintf('=========================================\n');


output_file = sprintf('dl_sumrate_data_1000_20_%d_20.mat', num_AP);
save(output_file,'betas', 'Gammas', 'Phii_cf', 'R_equal', 'R_frac', 'R_log', 'power');

fprintf('--> Đã lưu dữ liệu thành công vào file: %s\n', output_file);