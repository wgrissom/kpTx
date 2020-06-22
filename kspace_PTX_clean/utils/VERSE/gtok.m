function k = gtok(g,dt,gamma,numInteg)

%GTOK Gradient to k-space
%  This returns k-space for gradient
%
%  k = gtok(g,dt,gamma,numInteg)
%
%
%  written by Daeho Lee, 2009
%  Copyright (c) 2009 The Board of Trustees of The Leland Stanford Junior University. All Rights Reserved.


if (nargin < 4)
  numInteg = 'trapz';
end

switch numInteg 
case 'sum'
  k = dt*gamma*cumsum(g,1);
case 'trapz'
  k = dt*gamma*cumtrapz(g,1);
case 'simpson'
  k = dt*gamma*cumSimpsonUD(g,1);
otherwise
  disp('gtok: wrong method');
end

