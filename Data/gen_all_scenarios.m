function gen_all_scenarios(scenario_indices,varargin)
%GEN_ALL_SCENARIOS Generate/resume the two-pass data-generation batch.
% gen_all_scenarios() runs all 17; gen_all_scenarios(I) runs indices I.
% gen_all_scenarios(I,'num_sam_override',N) uses N samples per selected
% scenario, for end-to-end smoke testing without editing the scenario table.
if nargin < 1 || isempty(scenario_indices), scenario_indices = 1:17; end
scenarios = [6 30 2000;6 50 1000;6 100 1000;10 30 1000;10 50 1000;...
 10 100 1000;15 30 1000;15 50 1000;15 100 1000;6 20 500;6 40 500;...
 6 60 500;6 80 500;4 30 500;8 30 500;12 30 500;16 30 500]; % K,M,N
validateattributes(scenario_indices,{'numeric'},{'vector','integer','>=',1,'<=',17});
scenario_indices = scenario_indices(:)';
ip=inputParser;
addParameter(ip,'num_sam_override',[],@(x)isempty(x) || ...
 isnumeric(x) && isscalar(x) && isfinite(x) && x>=1 && fix(x)==x);
parse(ip,varargin{:}); num_sam_override=ip.Results.num_sam_override;
if ~isempty(num_sam_override), scenarios(scenario_indices,3)=num_sam_override; end

num_antenna=1; num_sr=3; tau=20; power_f=0.2; D=1; f=1900;
Hb=15; Hm=1.65; d0=0.01; d1=0.05; max_iter=50;
placeholder_nu=1; pool_size=12; chunk_size=100; pct=[50 75 90 95 99];
data_dir=fileparts(mfilename('fullpath')); out_dir=fullfile(data_dir,'New');
if ~isfolder(out_dir), mkdir(out_dir); end
stamp=datestr(now,'yyyymmdd_HHMMSS'); log_file=fullfile(out_dir,['gen_log_' stamp '.txt']);
diary(log_file); diary_guard=onCleanup(@()diary('off')); %#ok<NASGU>
nu_counts=ones(17,1); nu_counts(1)=5;
total_work=sum(nu_counts(scenario_indices).*scenarios(scenario_indices,3));
done_total=0; run_t=tic;
recent_n=[]; recent_t=[];
fprintf('\n============================================================\n');
fprintf('GEN_ALL_SCENARIOS | %s\n',datestr(now,31));
fprintf('Scenarios: %d | Pass-B realizations: %d | Pool: %d workers\n',...
 numel(scenario_indices),total_work,pool_size);
if ~isempty(num_sam_override)
 fprintf('SMOKE OVERRIDE: num_sam=%d for every selected scenario\n',num_sam_override);
end
fprintf('Log: %s\n============================================================\n',log_file);

p=gcp('nocreate');
if ~isempty(p) && (~isa(p,'parallel.ProcessPool') || p.NumWorkers~=pool_size)
 delete(p); p=[];
end
if isempty(p), p=parpool('Processes',pool_size); end
pool_guard=onCleanup(@close_pool); %#ok<NASGU>
manifest=fullfile(out_dir,'scenario_manifest.csv'); halted=false;

for idx=scenario_indices
 K=scenarios(idx,1); M=scenarios(idx,2); N=scenarios(idx,3); seed=1712+idx;
 tag=sprintf('%d_%d_%d',N,K,M);
 filename=['dl_isac_sumrate_data_' tag '.mat']; final=fullfile(out_dir,filename);
 cache=fullfile(out_dir,['passA_' tag '.mat']);
 partial=fullfile(out_dir,['partial_' tag '.mat']); scen_t=tic; last_done_index=0;
 if complete_final(final)
  fprintf('[scen %d/17 K=%d M=%d] SKIP complete: %s\n',idx,K,M,final);
  done_total=done_total+nu_counts(idx)*N; continue
 end
 fprintf('\n[scen %d/17 K=%d M=%d] START N=%d seed=%d at %s\n',...
  idx,K,M,N,seed,datestr(now,31));
 try
  % A partial is meaningful only with the exact channels in its Pass-A cache.
  % Never regenerate Pass A when a partial exists but its cache is missing.
  if isfile(partial) && ~isfile(cache)
   error('gen_all_scenarios:MissingPassACache',...
    'Partial exists but Pass-A cache is missing; refusing to regenerate channels: %s',cache);
  end
  if isfile(cache)
   fprintf('[scen %d/17 K=%d M=%d] Loading Pass-A cache\n',idx,K,M); A=load(cache);
  else
   rng(seed,'twister'); fprintf('[scen %d/17 K=%d M=%d] Pass A\n',idx,K,M);
   [betas,Gammas,Phii_cf,R_equal,R_frac,~,power_eq,power_frac,~,...
    rcs_values,ap_locations,sr_locations,q_a_all,q_b_all,q_c_all]=...
    downlink_ISAC_sumrate_data(N,M,K,num_sr,num_antenna,tau,power_f,...
     Hb,Hm,f,d0,d1,D,placeholder_nu,false);
   [sigma2_eq,fim_denom,g1]=sigma_equal(Gammas,power_eq,q_a_all,q_b_all,q_c_all);
   finite_sigma=sigma2_eq(isfinite(sigma2_eq));
   if isempty(finite_sigma), error('gen_all_scenarios:NoFiniteSigma','No finite sigma^2.'); end
   nu_vec=prctile(finite_sigma,pct); nu_main=nu_vec(4);
   save_cache(cache,betas,Gammas,Phii_cf,R_equal,R_frac,power_eq,power_frac,...
    rcs_values,ap_locations,sr_locations,q_a_all,q_b_all,q_c_all,...
    sigma2_eq,fim_denom,g1,nu_vec,nu_main); A=load(cache);
  end
  assert(isequal(size(A.Gammas),[N M K]),'Pass-A cache dimensions do not match.');
  if A.g1>1e-10
   halt_banner(idx,K,M,sprintf('G1 %.6g exceeds 1e-10',A.g1)); halted=true; break
  end

  nu_count=nu_counts(idx);
  if nu_count==5, solve_nu_indices=1:5; else, solve_nu_indices=4; end
  power_log_all=zeros(N,M,K,nu_count); R_log_all=zeros(nu_count,N);
  status_all=zeros(N,nu_count,'uint8'); converged_all=false(N,nu_count);
  if isfile(partial)
   P=load(partial);
   if ~valid_partial(P,N,M,K,nu_count,A.nu_vec)
    error('gen_all_scenarios:BadPartial','Incompatible partial: %s',partial);
   end
   power_log_all=P.power_log_all; R_log_all=P.R_log_all;
   status_all=P.status_all; converged_all=P.converged_all;
   last_done_index=P.last_done_index;
   fprintf('[scen %d/17 K=%d M=%d] RESUME solve %d/%d (Pass A loaded from cache)\n',...
    idx,K,M,last_done_index+1,nu_count*N);
  end
  done_total=done_total+last_done_index;
  while last_done_index<nu_count*N
   sweep_i=floor(last_done_index/N)+1; nu_i=solve_nu_indices(sweep_i);
   first=mod(last_done_index,N)+1;
   last=min(N,first+chunk_size-1); ids=first:last; bt=tic;
   [pb,rb,sb,cb]=solve_logapprox_batch(A.betas(ids,:,:),A.Gammas(ids,:,:),...
    A.Phii_cf(ids,:,:),A.q_a_all(ids,:,:),A.q_b_all(ids,:,:),A.q_c_all(ids,:,:),...
    A.nu_vec(nu_i),power_f,num_antenna,max_iter);
   power_log_all(ids,:,:,sweep_i)=pb; R_log_all(sweep_i,ids)=rb;
   status_all(ids,sweep_i)=sb; converged_all(ids,sweep_i)=cb;
   last_done_index=(sweep_i-1)*N+last; timestamp=datestr(now,31);
   save_partial(partial,power_log_all,R_log_all,status_all,converged_all,...
    last_done_index,A.nu_vec,timestamp);
   secs=toc(bt); n=numel(ids); done_total=done_total+n;
   recent_n(end+1)=n; recent_t(end+1)=secs; %#ok<AGROW>
   if numel(recent_n)>10, recent_n(1)=[]; recent_t(1)=[]; end
   rate=sum(recent_n)/sum(recent_t); eta=max(0,total_work-done_total)/rate;
   fprintf(['[scen %d/17 K=%d M=%d] nu %d/5 | %d/%d | %.1f%% total | '...
    'elapsed %s | ETA %s | %.2f s/real\n'],idx,K,M,nu_i,last,N,...
    100*done_total/total_work,hms(toc(run_t)),hms(eta),1/rate);
  end

  singular_pct=100*mean(A.fim_denom<=0);
  if nu_count==5, main_slice=4; else, main_slice=1; end
  main=status_all(:,main_slice);
  solved_pct=100*mean(main==0); infeasible_pct=100*mean(main==1);
  fallback_pct=100*mean(main==2); power_log=power_log_all(:,:,:,main_slice);
  R_log=R_log_all(main_slice,:); equal_rate=mean(A.R_equal); log_rate=mean(R_log);
  wallclock_sec=toc(scen_t); sigma_pct=prctile(A.sigma2_eq(isfinite(A.sigma2_eq)),pct);
  gates(A.g1,singular_pct,sigma_pct,A.nu_vec,A.nu_main,solved_pct,...
   infeasible_pct,fallback_pct,equal_rate,log_rate,wallclock_sec);
  if idx==1 && N==2000
   full_pct_levels=[5 10 25 50 75 90 95 99];
   full_sigma_pct=prctile(A.sigma2_eq(isfinite(A.sigma2_eq)),full_pct_levels);
   fprintf('Scenario 1 sigma^2 percentiles [5 10 25 50 75 90 95 99]: %s\n',...
    mat2str(full_sigma_pct,12));
  end
  if fallback_pct>1
   halt_banner(idx,K,M,sprintf('status=2 fallback %.3f%% exceeds 1%%',fallback_pct));
   halted=true; break
  end
  S=A; S.R_log=R_log; S.power_log=power_log; S.power_log_all=power_log_all;
  S.R_log_all=R_log_all; S.status_all=status_all; S.converged_all=converged_all;
  S.num_sr=num_sr; S.num_antenna=num_antenna; S.seed=seed; S.max_iter=max_iter;
  S.nu_count=nu_count;
  S=rmfield(S,intersect(fieldnames(S),{'fim_denom','g1'})); save_struct(final,S);
  append_manifest(manifest,idx,K,M,num_sr,N,A.nu_main,A.nu_vec,singular_pct,...
   solved_pct,infeasible_pct,fallback_pct,equal_rate,log_rate,max_iter,seed,...
   wallclock_sec,filename);
  if isfile(partial), delete(partial); end
  if isfile(cache), delete(cache); end
  fprintf('[scen %d/17 K=%d M=%d] COMPLETE saved %s\n',idx,K,M,final);
 catch ME
  fprintf(2,'\n*** SCENARIO ERROR idx=%d K=%d M=%d ***\n%s\nContinuing; caches retained.\n',...
   idx,K,M,getReport(ME,'extended','hyperlinks','off'));
  if exist('power_log_all','var') && exist('A','var')
   timestamp=datestr(now,31);
   try
    save_partial(partial,power_log_all,R_log_all,status_all,converged_all,...
     last_done_index,A.nu_vec,timestamp);
   catch CE, fprintf(2,'Emergency checkpoint failed: %s\n',CE.message); end
  end
 end
 clear A P power_log_all R_log_all status_all converged_all
end
if halted
 fprintf(2,'BATCH HALTED BY QUALITY GATE. Remaining scenarios were not run.\n');
else
 fprintf('\nGEN_ALL_SCENARIOS END %s | elapsed %s\n',datestr(now,31),hms(toc(run_t)));
end
end

function close_pool
p=gcp('nocreate'); if ~isempty(p), delete(p); end
end

function [s,d,g]=sigma_equal(G,P,qa,qb,qc)
N=size(G,1); M=size(G,2); s=inf(N,1); d=zeros(N,1); g=0;
for i=1:N
 p=sum(reshape(G(i,:,:),M,[]).*reshape(P(i,:,:),M,[]),2);
 g=max(g,max(abs(p-1))); a=reshape(qa(i,:,:),M,1);
 b=reshape(qb(i,:,:),M,1); c=reshape(qc(i,:,:),M,1);
 d(i)=(a'*p)*(b'*p)-(c'*p)^2;
 if d(i)>0, s(i)=((a+b)'*p)/d(i); end
end
end

function tf=complete_final(file)
tf=false; if ~isfile(file), return; end
req={'betas','Gammas','Phii_cf','R_equal','R_frac','R_log','rcs_values',...
 'ap_locations','sr_locations','q_a_all','q_b_all','q_c_all','power_eq',...
 'power_frac','power_log','nu_vec','nu_main','power_log_all','R_log_all',...
 'status_all','converged_all','num_sr','num_antenna','seed','max_iter','sigma2_eq','nu_count'};
try, names={whos('-file',file).name}; tf=all(ismember(req,names)); catch, tf=false; end
end

function tf=valid_partial(P,N,M,K,nu_count,nu)
req={'power_log_all','R_log_all','status_all','converged_all','last_done_index','nu_vec','timestamp'};
tf=all(isfield(P,req)) && size(P.power_log_all,1)==N &&...
 size(P.power_log_all,2)==M && size(P.power_log_all,3)==K &&...
 size(P.power_log_all,4)==nu_count && isequal(size(P.R_log_all),[nu_count N]) &&...
 isequal(size(P.status_all),[N nu_count]) &&...
 isequal(size(P.converged_all),[N nu_count]) && isequal(P.nu_vec,nu) &&...
 P.last_done_index>=0 && P.last_done_index<=nu_count*N;
end

function save_cache(file,betas,Gammas,Phii_cf,R_equal,R_frac,power_eq,...
 power_frac,rcs_values,ap_locations,sr_locations,q_a_all,q_b_all,q_c_all,...
 sigma2_eq,fim_denom,g1,nu_vec,nu_main)
tmp=[file '.tmp']; save(tmp,'-v7.3','betas','Gammas','Phii_cf','R_equal','R_frac',...
 'power_eq','power_frac','rcs_values','ap_locations','sr_locations','q_a_all',...
 'q_b_all','q_c_all','sigma2_eq','fim_denom','g1','nu_vec','nu_main'); movefile(tmp,file,'f');
end

function save_partial(file,power_log_all,R_log_all,status_all,converged_all,...
 last_done_index,nu_vec,timestamp)
tmp=[file '.tmp']; save(tmp,'-v7.3','power_log_all','R_log_all','status_all',...
 'converged_all','last_done_index','nu_vec','timestamp'); movefile(tmp,file,'f');
end

function save_struct(file,S)
tmp=[file '.tmp']; save(tmp,'-struct','S','-v7.3'); movefile(tmp,file,'f');
end

function append_manifest(file,idx,K,M,T,N,nu_main,nu,sing,solved,infeas,...
 fallback,equal_rate,log_rate,max_iter,seed,wall,filename)
fresh=~isfile(file); fid=fopen(file,'a'); assert(fid>=0,'Cannot append manifest.');
c=onCleanup(@()fclose(fid)); %#ok<NASGU>
if fresh
 fprintf(fid,['idx,K,M,T,num_sam,nu_main,nu_p50,nu_p75,nu_p90,nu_p95,nu_p99,'...
  'singular_pct,solved_pct,infeasible_pct,fallback_pct,equal_rate,log_rate,'...
  'max_iter,seed,wallclock_sec,filename\n']);
end
fprintf(fid,['%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,'...
 '%.9g,%.9g,%.9g,%.9g,%.17g,%.17g,%d,%d,%.3f,%s\n'],idx,K,M,T,N,...
 nu_main,nu,sing,solved,infeas,fallback,equal_rate,log_rate,max_iter,seed,wall,filename);
end

function gates(g1,sing,sp,nu,nm,solved,infeas,fallback,eq,lr,wall)
fprintf('G1 max|g_m - 1|: %.17g (expect ~1e-16)\n',g1);
fprintf('G2 fraction denom <= 0: %.6f%% (expect < 1%%)\n',sing);
fprintf('G3 sigma^2 percentiles [50 75 90 95 99]: %s\n',mat2str(sp,12));
fprintf('   nu_vec: %s | nu_main: %.12g\n',mat2str(nu,12),nm);
fprintf('G4 status at nu_main: solved %.3f%% / infeasible %.3f%% / fallback %.3f%%\n',...
 solved,infeas,fallback);
fprintf('G5 mean sum-rate: Equal Power %.12g | Log-approx %.12g\n',eq,lr);
fprintf('G6 scenario wall-clock: %s (%.3f s)\n',hms(wall),wall);
end

function halt_banner(idx,K,M,reason)
fprintf(2,['\n!!!!!!!!!!!!!!!! QUALITY-GATE HALT !!!!!!!!!!!!!!!!\n'...
 'Scenario idx=%d K=%d M=%d failed: %s\nInspect Pass-A cache, partial, solver status, and diary.\n'...
 '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'],idx,K,M,reason);
end

function s=hms(x)
if ~isfinite(x), s='--:--:--'; return; end
x=max(0,round(x)); s=sprintf('%02d:%02d:%02d',floor(x/3600),floor(mod(x,3600)/60),mod(x,60));
end
