function [b1v,gv] = toVERSE(b1,g,dt, b1gBound, sMax,gMax,gamma, dispOn,...
  osMtg, gInit,gFin, numInteg, osIn,osOut)

%toVERSE Time-Optimal VERSE 
%  This returns time-optimal post-VERSE waveform pair {b1v,gv} for pre-VERSE pair {b1,g}
%  constrained by either peak RF power or gradient upperbound prescribed by 'b1gBound'.
%
%  This is a wrapping function for 'minTimeVERSE' to facilitate fidelity control.
%  Example: 'osMtg = 1' for fast calculation and 'osMtg = 4' for better fidelity.
%
%  [b1v,gv] = toVERSE(b1,g,dt, b1gBound, sMax,gMax,gamma,...
%    gInit,gFin, numInteg, osIn,osOut,osMtg)
%
%  INPUT:   
%    b1		-  pre-VERSE complex RF matrix NtxNc [G]
%		   must begin and end with zero amplitude
%    g		-  pre-VERSE real graident matrix NtxNd [G/cm]
%                  Nt >= 3, Nd = {1,2,3}
%    dt		-  waveform sampling time (sec)
%    b1gBound	-  either scalar b1Bound or vector gBound(t)
%                  arbitrary sampling time (other than dt) is permitted for gBound(t)
%    sMax	-  maximum gradient slew rate [G/cm/s]
%    gMax	-  maximum gradient amplitude [G/cm]
%    gamma	-  gyromagnetic ratio [Hz/G]
%    dispOn	-  show plots
%    osMtg	-  oversampling factor for time-optimal gradient design
%    gInit	-  initial gradient amplitude
%                  Leave empty for gInit = 0.
%    gFin	-  gradient value at the end of the trajectory
%                  If not possible, the result would be the largest possible ampltude.
%                  Leave empty if you don't care to get maximum gradient.
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
%    [b1v,gv] = toVERSE(b1,g,dt,b1Bound,sMax,gMax,gamma);
%
%    % Ex 2: limiting maximum gradient amplitude
%    gBound = 0.5*gMax*ones(size(g,1),1);
%    [b1v,gv] = toVERSE(b1,g,dt,gBound,sMax,gMax,gamma);
%
%
%  written by Daeho Lee, 2009
%  Copyright (c) 2009 The Board of Trustees of The Leland Stanford Junior University. All Rights Reserved.


%% Default Parameters
if (nargin < 8) | isempty(dispOn)
  dispOn = false;
end
if (nargin < 9) | isempty(osMtg)
  if size(g,2) > 1
    osMtg = 4;  % for fidelity
  else
    osMtg = 1;  % for fast calculation
  end
end
if (nargin < 10) | isempty(gInit)
  gInit = 0;
end
if (nargin < 11) | isempty(gFin)
  gFin = 0;
end
if (nargin < 12) | isempty(numInteg)
  numInteg = 'trapz';
end
if (nargin < 13) | isempty(osIn)
  osIn = 32;
end
if (nargin < 14) | isempty(osOut)
  osOut = 32;
end
debug_on = false;


%% Input Checks
if ~isPosIntNum(osMtg) | (osMtg > osOut)
  error('wrong mtg os fact');
end
if mod(osOut,osMtg) ~= 0
  error('osOut must be divisible by osMtg');
end


%% Main Routine 
if osMtg == 1
  [b1v,gv] = minTimeVERSE(b1,g,dt, b1gBound, sMax,gMax,gamma, dispOn,...
    gInit,gFin,dt,numInteg,osIn,osOut);
else
  [b1v,gv] = minTimeVERSE(b1,g,dt, b1gBound, sMax,gMax,gamma, dispOn,...
    gInit,gFin,dt/osMtg,numInteg,osIn,osOut/osMtg);
  b1v = b1v(1:osMtg:end,:);
  gv = gv(1:osMtg:end,:);

  % PostCondition: enforcing initial/final values
  g_thresh = sMax*dt/1000;
  if gInit == 0 
    if (absv(gv(1,:),2) > g_thresh)
      gv = [zeros(1,size(gv,2));g];
      b1v = [zeros(1,size(b1v,2));b1]; 
    else
      gv(1,:) = 0;
      b1v(1,:) = 0;
    end
  end
  if gFin == 0
    if (absv(gv(end,:),2) > g_thresh)
      gv = [gv;zeros(1,size(gv,2))];
      b1v = [b1v;zeros(1,size(b1v,2))]; 
    else
      gv(end,:) = 0;
      b1v(end,:) = 0;
    end
  end
end

