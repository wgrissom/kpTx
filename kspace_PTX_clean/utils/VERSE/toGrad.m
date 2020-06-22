function g = toGrad(k,gInit,gFin, sOfGMax,gMax, sMax,gamma,dt)

%toGrad Time-Optimal Time-Optimal Gradient Design 
%  This returns time-optimal post-VERSE waveform pair {b1v,gv} for pre-VERSE pair {b1,g}
%  constrained by either peak RF power or gradient upperbound prescribed by 'b1gBound'.
%
%  This is a wrapping function for 'minTimeGrad' for consistent use of gradient matrix
%  regardless of the dimension.
%
%  g = toGrad(k,gInit,gFin, sOfGmax,gMax, sMax,gamma,dt)
%
%  INPUT:   
%    k		-  k-space NtxNd matrix [1/cm], Nd = {1,2,3}
%    gInit	-  initial gradient amplitude
%                  Leave empty for gInit = 0.
%    gFin	-  gradient value at the end of the trajectory
%                  If not possible, the result would be the largest possible ampltude.
%                  Leave empty if you don't care to get maximum gradient.
%    s		-  interpolated arc-length for gMax vector
%    sMax	-  maximum gradient slew rate [G/cm/s]
%    gMax	-  maximum gradient amplitude [G/cm]
%    gamma	-  gyromagnetic ratio [kHz/G]
%    dt		-  waveform sampling time [sec]
%
%  OUTPUT: 
%    g		-  real graident matrix, sampled at dt [G/cm]
%
%
%  written by Daeho Lee, 2009
%  Copyright (c) 2009 The Board of Trustees of The Leland Stanford Junior University. All Rights Reserved.


Ng = size(k,2);
 
switch Ng 
case 1
  k = i*k;  
case 2
  k = k(:,1) + i*k(:,2);
case 3
otherwise
  error('wrong dimension');
end

g = minTimeGrad(k,gInit,gFin, sOfGMax,gMax, sMax*1e-3,gamma*1e-3,dt*1e+3,0);

switch Ng
case 1
  g = imag(g);  
case 2
  g = [real(g),imag(g)];
case 3
otherwise
  error('wrong dimension');
end

