function y = isPosIntNum(x)
% returns true if x is a positive integer number

%ISPOSINTNUM
%  This returns true if x is a positive integer number.
%
%  y = isPosIntNum(x)
%
%
%  written by Daeho Lee, 2009
%  Copyright (c) 2009 The Board of Trustees of The Leland Stanford Junior University. All Rights Reserved.


y = (x > 0) & (ceil(x) == floor(x));

