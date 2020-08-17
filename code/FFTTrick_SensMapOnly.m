% calib is in image domain
function [F_c, dgrid] = FFTTrick_SensMapOnly(calib, nComb,densfact)
% F_c are cells of Hadamard product between conj(calib) and calib
[nd, calibh] = deal(size(calib), conj(calib));
%if nd(3)==1,[nc,nd,pd]=deal(nd(4),nd(1:2),[128,128]);
%else, [nc,nd,pd]=deal(nd(4),nd(1:3),[128,128,128]);
if nd(3)==1,[nc,nd]=deal(nd(4),nd(1:2));
else, [nc,nd]=deal(nd(4),nd(1:3));
end % pd set empirically
pd=nd*densfact; 

[ich, ic] = ndgrid(1:nc);
ltMask = ich>=ic; % lower triangle mask
coilSub = [ich(ltMask), ic(ltMask)];

dgrid = (nd-1)./(pd-1); % unit distance of the padded Cartesian grid

F_c = cell(1, nc);
calib0 = zeros(pd, class(calib));

ncSub = ctrSub(nd);
fn1 = @(ii)mod((1:nd(ii))-ncSub(ii), pd(ii))+1; % inds after proper 0-padding
s_c = cellfun(fn1, num2cell(1:numel(nd)), 'Uni', false); % cell of subscripts
% s_c is the indeces of the points that will be subsituded in the padded matrix
% If calib is 20*20, then s_c are the 10 right most and the 10 left most points 
% this matches the fft assumption where t=0 (middle point) is at the left most
% note that s_c is the right most then the left most.
% (lower and upper are similar)

% calib has low image domain resolution corresponding to the low kspace FOV of ACS
% calib0 still had the same low image domain resolution, but has larger image domain FOV (zero padding)
% There for F_c=FFT(calib0), still had the low kpsace FOV, but much denser sampling 

kshift = ctrSub(pd)-1;
for ii = 1:nc % huge FFT is too slow in parfor
%for ii = 1:nc %Don't need combination
  %chc_i = calibh(:,:,:,coilSub(ii,1)).*calib(:,:,:,coilSub(ii,2));
  chc_i = calib(:,:,:,coilSub(ii,1));
  % trick: replacing 0-field is faster than proper 0-padding
  calib0(s_c{:}) = chc_i;
  % One may replace the circshift below by querying w/ modded locs, while it may
  % shorten the time of this for-loop, the moding operation may aggravate the
  % cost of the enormous query operations; Also, the modded query only works for
  % interpolation methods that strictly do not involve out-of-bndry values.
  
  F_c{ii} = circshift(fft3(calib0), kshift);
end

end
