function y = plot3d(xm,xl,yl,zl)

%PLOT3D Plot matrix upto 3D 
%  This plots matrix according to the dimension of input xm. 
%
%  y = plot3d(xm,xl,yl,zl)
%
%
%  written by Daeho Lee, 2009
%  Copyright (c) 2009 The Board of Trustees of The Leland Stanford Junior University. All Rights Reserved.


if nargin < 2
  xl = 'x';
end
if nargin < 3
  xl = 'y';
end
if nargin < 4
  xl = 'z';
end

dim = size(xm,2);
switch dim
case 1
  plot(xm); xlabel('n'); ylabel(xl);
case 2
  plot(xm(:,1),xm(:,2)); xlabel(xl); ylabel(yl);
case 3
  plot3(xm(:,1),xm(:,2),xm(:,3)); xlabel(xl); ylabel(yl); zlabel(zl);
otherwise
  y = false;
  error('supports upto 3 dimensions');
end

y = true;
