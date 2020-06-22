function s = gtos(g,dt,gamma,numInteg)

%GTOS Gradient to arc-length
%  This returns arc-length for gradient
%
%  k = gtos(g,dt,gamma,numInteg)
%
%
%  written by Daeho Lee, 2009
%  Copyright (c) 2009 The Board of Trustees of The Leland Stanford Junior University. All Rights Reserved.


if (nargin < 4)
  numInteg = 'trapz';
end

switch numInteg 
case 'sum'
  if absv(g,1) == 0
    g(end) = eps;
  end
  s = dt*gamma*cumsum(absv(g,2));
case 'trapz'
  s = dt*gamma*cumtrapz(absv(g,2));
case 'simpson'
  s = dt*gamma*cumSimpsonUD(absv(g,2));
otherwise
  disp('gtos: wrong method');
end

