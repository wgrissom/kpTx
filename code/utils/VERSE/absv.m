function y = absv(x,dim)

%ABSV Absolute value, vector version
%  This returns l2-norm of vector along the dimension DIM.
%
%  y = absv(x,dim)
%
%
%  written by Daeho Lee, 2009
%  Copyright (c) 2009 The Board of Trustees of The Leland Stanford Junior University. All Rights Reserved.


y = sqrt(sum(abs(x).^2,dim));
 
