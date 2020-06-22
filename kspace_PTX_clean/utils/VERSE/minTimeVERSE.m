function [b1_v,g_v] = minTimeVERSE(b1,g,dt, b1gBound, sMax,gMax,gamma, dispOn,...
  gInit,gFin,dt_v,numInteg,osIn,osOut)

%minTimeVERSE Time-Optimal VERSE 
%  This returns time-optimal post-VERSE waveform pair {b1v,gv} for pre-VERSE pair {b1,g}
%  constrained by either peak RF power or gradient upperbound prescribed by 'b1gBound'.
%
%  [b1_v,g_v] = minTimeVERSE(b1,g,dt, b1gBound, sMax,gMax,gamma,...
%    gInit,gFin,dt_v,numInteg,osIn,osOut,dispOn,debug_on)
%
%  INPUT:   
%    b1		-  pre-VERSE complex RF matrix NtxNc [G]
%		   must begin and end with zero amplitude
%    g		-  pre-VERSE real graident matrix NtxNd [G/cm]
%                  Nt >= 3, Nd = {1,2,3}
%		   Currently, must be non-zero except initial and final samples.
%    dt		-  input waveform sampling time [sec]
%    b1gBound	-  either scalar b1Bound or vector gBound(t)
%                  arbitrary sampling time (other than dt) is permitted for gBound(t)
%    sMax	-  maximum gradient slew rate [G/cm/s]
%    gMax	-  maximum gradient amplitude [G/cm]
%    gamma	-  gyromagnetic ratio [kHz/G]
%    dispOn	-  show plots
%    gInit	-  initial gradient amplitude
%                  Leave empty for gInit = 0.
%    gFin	-  gradient value at the end of the trajectory
%                  If not possible, the result would be the largest possible ampltude.
%                  Leave empty if you don't care to get maximum gradient.
%    dt_v	-  output waveform sampling time [sec]
%    numInteg	-  numerical integration method {'sum','trapz','simpson'}
%    osIn	-  oversampling factor for pre-VERSE
%    osOut	-  oversampling factor for arc-length calculation on post-VERSE 
%
%  OUTPUT: 
%    b1v	-  post-VERSE complex RF matrix [G]
%    gv		-  post-VERSE real grdient matrix [G/cm]
%
%
%  USAGE:
%
%    gamma = 4257;
%    sMax = 15000;
%    gMax = 4;
%    dt = 4e-6;
%
%    % Ex 1: limiting peak RF power
%    b1Bound = 0.5*max(abs(b1(:)));
%    [b1v,gv] = minTimeVERSE(b1,g,dt,b1Bound,sMax,gMax,gamma);
%
%    % Ex 2: limiting maximum gradient amplitude
%    gBound = 0.5*gMax*ones(size(g,1),1);
%    [b1v,gv] = minTimeVERSE(b1,g,dt,gBound,sMax,gMax,gamma);
%
%
%  written by Daeho Lee, 2009
%  Copyright (c) 2009 The Board of Trustees of The Leland Stanford Junior University. All Rights Reserved.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Default Parameters 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargin < 8) | isempty(dispOn)
  dispOn = false;
end
if (nargin < 9) | isempty(gInit)
  gInit = 0;
end
if (nargin < 10) | isempty(gFin)
  gFin = 0;
end
if (nargin < 11) | isempty(dt_v)
  dt_v = dt;
end
if (nargin < 12) | isempty(numInteg)
  numInteg = 'trapz'; % {'sum','trapz','simpson'}
end
if (nargin < 13) | isempty(osIn)
  osIn = 32;
end
if (nargin < 14) | isempty(osOut)
  osOut = osIn*round(dt_v/dt);
end
debug_on = false;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Input Checks
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if absv(b1(1,:),2) > eps 
  error('initial b1 must be zero');
else
  b1(1,:) = 0;
end
if absv(b1(end,:),2) > eps 
  error('final b1 must be zero');
else
  b1(end,:) = 0;
end
if dt < 0
  error('wrong dt');
end
if dt_v < 0
  error('wrong dt\_v');
end
if ~isempty(find(b1gBound < 0)) 
  error('wrong b1gBound');
end
if sMax <= 0
  error('wrong smax');
end
if gMax <= 0
  error('wrong smax');
end
if gamma <= 0
  error('wrong gamma');
end
if (gInit ~= 0) & ~isempty(gInit)
  error('wrong gInit');
end
if (gFin ~= 0) & ~isempty(gFin)
  error('wrong gFin');
end
if ~isPosIntNum(osIn)
  error('wrong os fact in');
end
if ~isPosIntNum(osOut)
  error('wrong os fact in');
end

[Nt,Nd] = size(b1gBound);
if (Nd ~= 1) | (Nt < 1)
  error('wrong b1gBound');
end

% TODO: how to accomodate zero gradient...
%       e.g. g(find(absv(g,2) == 0),:) = 1e-10;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Problem to Solve 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Nt == 1
  disp(['.... designing time-optimal VERSE for B1max = ',num2str(b1gBound)]);
  verse_mode = 'b1Bound';
  b1Bound = b1gBound;
elseif Nt >= 3
  disp('.... designing VERSE for given Gout');
  verse_mode = 'gBound';
  gBound = b1gBound;
else
  error('.... wrong b1gBound');
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Load RF and G
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Nt, Nc] = size(b1);
k = gtok(g,dt,gamma,numInteg);

%% enforcing zero RF for zero gradient
b1(find(absv(g,2) < eps),:) = 0; 






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% 1. Routine Controls 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
interp_b1 = 'spline';
interp_g = 'spline';
interp_Ws = 'spline';

disp(['.... using oversampling factor of ',num2str(osIn),' for input waveforms in s-domain']); 
disp(['.... using oversampling factor of ',num2str(osOut),' for output waveforms in s-domain']);
disp(['.... using numerical integration method: ',numInteg]);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Input Oversampling
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if osIn > 1
  dt_os = dt/osIn;
  Nt_os = (Nt-1)*osIn+1;
  b1_os = interp1([1:Nt].',b1,linspace(1,Nt,Nt_os).',interp_b1);
  g_os = interp1([1:Nt].',g,linspace(1,Nt,Nt_os).',interp_g);
else
  dt_os = dt;
  Nt_os = Nt;
  b1_os = b1; 
  g_os = g;
end
k_os = gtok(g_os,dt_os,gamma,numInteg);
s_os = gtos(g_os,dt_os,gamma,numInteg);
Wg_os = zeros(size(b1_os));
idx_nz = find(absv(g_os,2) > eps);
Wg_os(idx_nz,:) = b1_os(idx_nz,:)./repmat((absv(g_os(idx_nz,:),2)),[1,Nc]);

% to check oversampling done ok
if debug_on
  figure;
  ncoil = 1;
  tidx = [0:Nt-1]*dt*1e+3;
  tidx_os = [0:Nt_os-1]*dt_os*1e+3; 
  subplot(2,1,1); plot(tidx,absv(g,2)); hold on; plot(tidx_os,absv(g_os,2),'r.'); axis tight;
    legend('original input','oversampled');
    xlabel('time [ms]'); ylabel('G [G/cm]'); title('Input Oversampling Test');
  subplot(2,1,2); plot(tidx,abs(b1(:,ncoil))); hold on; plot(tidx_os,abs(b1_os(:,ncoil)),'r.'); axis tight; ylabel('B_1 [G]');
    xlabel('time [ms]'); ylabel('B_1 [G]');
    legend('original input','oversampled');
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% dt-VERSE 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch verse_mode
case 'b1Bound'
  mu = b1Bound/gMax;
  eta_s = min(1,mu./(abs(Wg_os)+eps));
  gMaxAll = gMax*ones(Nt_os,Nc).*eta_s;
case 'gBound'
  s_tmp = gtos(gBound,dt,gamma,numInteg); % in fact, dt and gamma can be any number due to normalization afterwards
  s_tmp = s_tmp/s_tmp(end)*s_os(end);     % forcing equal arclength
  gMaxAll = interp1(s_tmp,gBound,s_os,interp_g);
  gMaxAll = repmat(gMaxAll,[1,Nc]);
otherwise
  error('wrong verse mode');
end
gMaxBound = min(gMaxAll,[],2);
% For numerical stability, gMaxBound must be non-zero
gMaxBound = max(gMaxBound,sMax*dt_v*ones(size(gMaxBound)));

tstart = tic;
g_v = toGrad(k_os,gInit,gFin,s_os,gMaxBound,sMax,gamma,dt_v);
telapsed = toc(tstart);
fprintf('.... Elapsed time for toGrad: %0.2f secs\n',telapsed);

% PostCondition: enforcing initial/final values
gInitFinMin = sMax*dt_v/1000;
if gInit == 0 
  if (absv(g_v(1,:),2) > gInitFinMin)
    g_v = [zeros(1,size(g_v,2));g_v];
  else
    g_v(1,:) = 0;
  end
end
if gFin == 0
  if (absv(g_v(end,:),2) > gInitFinMin)
    g_v = [g_v;zeros(1,size(g_v,2))];
  else
    g_v(end,:) = 0;
  end
end

Nt_v = size(g_v,1);
k_v = gtok(g_v,dt_v,gamma,numInteg);
sr_v = diff(g_v)/dt_v;


 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Output Oversampling
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate arclength on oversampled gradient
if osOut > 1
  g_tmp = interp1([1:Nt_v].',g_v,linspace(1,Nt_v,(Nt_v-1)*osOut+1).',interp_g);
  s_tmp = gtos(g_tmp,dt_v/osOut,gamma,numInteg);
  s_v = s_tmp(1:osOut:end);
else
  s_v = gtos(g_v,dt_v,gamma,numInteg);
end
% checking gradient area mismatch
s_err_norm = (s_v(end) - s_os(end))/s_os(end);
if abs(s_err_norm) > (s_os(end)/size(s_os,1))*1e-2
  disp(['.... NOTE: (Lv-Los)/Los = ',num2str(s_err_norm),': Los = ',num2str(s_os(end)),', Lv = ',num2str(s_v(end))]);
end
Wg_v = interp1(s_os,Wg_os,s_v,interp_Ws);
Wg_v(find(s_v>max(s_os)),:) = 0;
b1_v = Wg_v.*repmat(absv(g_v,2),[1,Nc]);
b1_v(1,:) = 0;
b1_v(end,:) = 0;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Results 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if dispOn
  figure;
  sr_os = diff(g_os)/dt_os;
  tidx = [1:Nt_os]*dt_os*1e+3;
  subplot(3,4,1); plot(tidx,abs(b1_os)); grid on; axis tight; xlabel('time [ms]'); ylabel('B_1 [G]'); title('S1: pre-VERSE in t-domain');
  subplot(3,4,5); plot(tidx,[absv(g_os,2),g_os]); grid on; axis tight; xlabel('time [ms]'); ylabel('G [G/cm]');
  subplot(3,4,9); plot(tidx(1:end-1),absv(sr_os,2)); grid on; axis tight; xlabel('time [ms]'); ylabel('|S| [G/cm/ms]');

  subplot(3,4,2); plot(s_os,absv(Wg_os,2)); grid on; axis tight; xlabel('s [1/cm]'); ylabel('|W_g|'); title('S2: pre-VERSE in s-domain');
  subplot(3,4,6); plot(s_os,absv(g_os,2)); grid on; axis tight;
    xlabel('s [1/cm]'); ylabel('|G| [G/cm]'); title('pre-VERSE');
  subplot(3,4,10); plot3d(k_os,'k_x [1/cm]','k_y [1/cm]','k_z [1/cm]'); grid on; axis tight;

  subplot(3,4,3); plot(s_v,absv(Wg_v,2)); grid on; axis tight; xlabel('s [1/cm]'); ylabel('|W_g|'); title('S3: post-VERSE in s-domain');
  subplot(3,4,7); plot(s_os,gMaxBound,'k','LineWidth',3); grid on; hold on; plot(s_v,absv(g_v,2)); axis tight;
    legend('G_u'); xlabel('s [1/cm]'); ylabel('|G| [G/cm]'); title('post-VERSE');
  subplot(3,4,11); plot3d(k_v,'k_x [1/cm]','k_y [1/cm]','k_z [1/cm]'); grid on; axis tight;

  tidx = [1:Nt_v]*dt_v*1e+3;
  subplot(3,4,4); plot(tidx,abs(b1_v)); grid on; axis tight; xlabel('time [ms]'); ylabel('B_1 [G]'); title('S4: post-VERSE in t-domain');
  subplot(3,4,8); plot(tidx,[absv(g_v,2),g_v]); grid on; axis tight; xlabel('time [ms]'); ylabel('G [G/cm]');
  subplot(3,4,12); plot(tidx(1:end-1),absv(sr_v,2)); grid on; axis tight; xlabel('time [ms]'); ylabel('|S| [G/cm/ms]');
end
