function W = kpTx(b1, k, k_pTx_args, offRes_args)

% k-Space Domain Parallel Transmit Pulse Design
%
% Solve for a sparse matrix W that relates the FT of a target excitation
% pattern to the parallel RF pulses that produce it.
%
% Inputs:
%   b1: Nx x Ny x Nz x nCoils B1+ map array
%   k:  Nt x 3 3D excitation k-space trajectory (cycles/fov)
%   k_pTx_args: (Optional) Structure with the following fields:
%   offRes_args: (Optional) Structure with the following fields:
%
% Output:
%   W: Nt * nCoils x Nx*Ny*Nz sparse matrix
%
% 2020 Martin Ma and William Grissom, Vanderbilt University

[dimb1(1), dimb1(2), dimb1(3), nCoils] = size(b1);

if ~exist('offRes_args', 'var')
    ifOffRes = false;
else
    if ~isempty(offRes_args)
        ifOffRes = true;
        b0 = offRes_args{1};
        mask = offRes_args{2};
        dt = offRes_args{3};
        Lseg = offRes_args{4};
    else
        ifOffRes = false;
    end
end

if ~exist('k_pTx_args', 'var')
    kWrapBack = false;
    segWidth = 4;
    nHood = 4;
    Tik = 1;
    nThreads = 1;
else
    segWidth = k_pTx_args{1}; % number of W matrix columns in each dimension to simultaneously solve for
    nHood = k_pTx_args{2}; % maximum distance from any target point in the simultaneous block to any included trajectory point
    Tik = k_pTx_args{3};
    kWrapBack = k_pTx_args{4};
    nThreads = k_pTx_args{5};
end

kposx = -dimb1(1)/2:dimb1(1)/2-1;
kposy = -dimb1(2)/2:dimb1(2)/2-1;
kposz = -dimb1(3)/2:dimb1(3)/2-1;
[kxPos, kyPos, kzPos] = meshgrid(kposx, kposy, kposz);  % in cycles/FOV

linearInds = reshape(1:dimb1(1)*dimb1(2)*dimb1(3), ...
    [dimb1(1) dimb1(2) dimb1(3)]);
[xI,yI,zI] = ndgrid(1:dimb1(1), 1:dimb1(2), 1:dimb1(3));

segCenters = linearInds(1:segWidth:end, 1:segWidth:end, 1:segWidth:end); % center indices of segments


kxTraj = k(:,1);
kyTraj = k(:,2);
kzTraj = k(:,3);


if kWrapBack
    % tile the trajectory around the center to find traj points circulantly
    kxTrajTile = zeros(length(kxTraj(:)),27);
    kyTrajTile = zeros(length(kyTraj(:)),27);
    kzTrajTile = zeros(length(kzTraj(:)),27);
    
    for ii = 1:3
        for jj = 1:3
            for kk=1:3
                kxTrajTile(:,(ii-1)*3+jj+(kk-1)*9) = kxTraj + (ii-2)*dimb1(1);
                kyTrajTile(:,(ii-1)*3+jj+(kk-1)*9) = kyTraj + (jj-2)*dimb1(2);
                kzTrajTile(:,(ii-1)*3+jj+(kk-1)*9) = kzTraj + (kk-2)*dimb1(3);
            end
        end
    end
    kxTrajTile = kxTrajTile(:);
    kyTrajTile = kyTrajTile(:);
    kzTrajTile = kzTrajTile(:);
else
    
    kxTrajTile = kxTraj;
    kyTrajTile = kyTraj;
    kzTrajTile = kzTraj;
    
end


linearIndsTile = repmat(linearInds,[3 3 3 ]);

%%
if ifOffRes
    
    tb0 = (0:size(k,1)-1)*dt/1000-size(k,1)*dt/1000;
    % call Michigan IRT function to generate time-segmented model
    [B, C] = mri_exp_approx(tb0, 1i*2*pi*b0(mask), Lseg, ...
		'type', {'hist,time,unif', 40});
    Ct = C.';
    
    % apply off-resonance correction
    % embed spatial interpolators to sensitivity maps
    CtFull = zeros([Lseg dimb1]);
    CtFull(:, mask) = Ct.';
    CtFull = permute(CtFull, [2 3 4 1]);
    BFull = B;
    
    BFull = single(BFull);
    BFull_c = cell(1);
    BFull_c{1} = BFull;
    
    calib = zeros([dimb1 nCoils Lseg]);
    for ii = 1:Lseg
        calib(:,:,:,:,ii) = bsxfun(@times, b1, CtFull(:, :, :, ii));
    end
    calib = calib(:, :, :, :);
    nCoilsLseg = nCoils*Lseg;
    
    disp 'Building FT maps'
    tic
    nComb = nCoilsLseg * (nCoilsLseg + 1) / 2;
    calib = single(calib);
    FToversamp = 1;   % Oversampling factor = 1 to save memory
    [F_c, ~] = FFTTrick(calib, nComb, FToversamp);
    
    toc
    
    [F_c_SensMapOnly, dgrid] = FFTTrick_SensMapOnly(calib, nComb, FToversamp);
    
    
    
else
    
    disp 'Building FT maps'
    tic
    nComb = nCoils*(nCoils+1)/2; % number of coil combinations
    calib = single(b1); % input B1 maps has to be single
    FToversamp=2;   % Oversampling factor of
    [F_c, ~] = FFTTrick(calib, nComb, FToversamp);
    toc
    
    [F_c_SensMapOnly, dgrid] = FFTTrick_SensMapOnly(calib, nComb, FToversamp);
    
    % F_c is the oversampled FT maps of the coil-combined spatial sensitivity
    % maps
    
    % F_c has to be cell of single matrices, due to the mxGetData and
    % mxGetImagData used in LS_fft_mex fucntion
    
    %  shift and kTrajInds have to be single for the same reason.
    
end

%%


shift_c_mex=cell(1,length(segCenters(:)));
kTrajInds_ii=cell(1,length(segCenters(:)));

for ii = 1:length(segCenters(:))
    
    centerInd = segCenters(ii);
    centerIndRow = xI(centerInd);
    centerIndCol = yI(centerInd);
    centerIndHih = zI(centerInd);
    
    
    % find nearby points on trajectory, which can influence target segment
    kTrajInds = find(kxTrajTile >= kxPos(centerInd)-nHood & ...
        kxTrajTile <= kxPos(centerInd)+segWidth-1+nHood & ...
        kyTrajTile >= kyPos(centerInd)-nHood & ...
        kyTrajTile <= kyPos(centerInd)+segWidth-1+nHood & ...
        kzTrajTile >= kzPos(centerInd)-nHood & ...
        kzTrajTile <= kzPos(centerInd)+segWidth-1+nHood);
    
    kTrajInds1=kTrajInds;
    kTrajInds_ii{ii} = single(mod(kTrajInds-1,length(kxTraj(:)))+1);
    %%% kTrajTile is for wrap back. and mod is to get rid of the index
    %%% increament (the sudden change) between wrap back.
    nNgb(ii)=length(kTrajInds1);
    
    shift_c_mex{ii} = single(-[kyTrajTile(kTrajInds1)-kyPos(centerIndRow,centerIndCol,centerIndHih), kxTrajTile(kTrajInds1)-kxPos(centerIndRow,centerIndCol,centerIndHih), kzTrajTile(kTrajInds1)-kzPos(centerIndRow,centerIndCol,centerIndHih)])';
    % cycle/FOV, shift_c_mex is the relative shifts against the first
    % point of the target segment. It is needed for the interpolation.
    % shift_c_mex has to be cell of single matrices due to data handling of mex function.
end




%%
idgrid = single(1./dgrid);

% Since shift_c_mex only carries ralative shifts of the excitation
% trajectory points.
% Here shSolve is needed to carry the relative shifts of the Target points
% within a segment.
% Also in cycle/FOV, and also relative to the first point of the segment.
[kySolve,kxSolve,kzSolve]=ndgrid(0:segWidth-1);
shSolve=-[kySolve(:) kxSolve(:) kzSolve(:)];
shSolve=single(shSolve);

% In this method, every individual column of W is solved with a same
% regularizer Tik
disp 'Solving for columns of W matrix'
tic
if ifOffRes
    if  nThreads ~=0
        w_coeffs = kpTx_solve_w_OpenMP(shift_c_mex, nCoils*Lseg, Tik, ...
            F_c, idgrid, shSolve', F_c_SensMapOnly, nCoils, BFull_c, ...
            kTrajInds_ii, nThreads);
    else
        w_coeffs = kpTx_solve_w(shift_c_mex, nCoils*Lseg, Tik, F_c, ...
            idgrid, shSolve', F_c_SensMapOnly,nCoils,BFull_c,kTrajInds_ii);
    end
else
    if  nThreads ~=0
        w_coeffs = kpTx_solve_w_OpenMP(shift_c_mex, nCoils, Tik, F_c, ...
            idgrid, shSolve', F_c_SensMapOnly, nThreads);
    else
        w_coeffs = kpTx_solve_w(shift_c_mex, nCoils, Tik, F_c, idgrid, ...
            shSolve', F_c_SensMapOnly);
    end
end
toc

%%
disp 'Filling W matrix'
tic
% get total number of non-zero entries - can this be done more efficiently,
% earlier in script?
totalnonzero = 0;
for ii = 1:length(segCenters(:)) % loop over sectors
    
    if ~isempty(shift_c_mex{ii})
        
        kTrajIndsAll = double(repmat(kTrajInds_ii{ii}(:),[1 nCoils]));
        for ll = 2:nCoils
            kTrajIndsAll(:,ll) = kTrajIndsAll(:,ll) + (ll-1)*length(kxTraj);
        end
        
        [i,j] = ndgrid(kTrajIndsAll(:),1:segWidth^3);
        Wentries{ii} = sparse(i, j, double(w_coeffs{ii}(:)), nCoils*length(kxTraj), segWidth^3);
        totalnonzero = totalnonzero + nCoils * length(kTrajInds_ii{ii}(:)) * segWidth^3;
        
    end
    
end

W = spalloc(length(kxTraj(:))*nCoils, dimb1(1)*dimb1(2)*dimb1(3), totalnonzero);
for ii = 1:length(segCenters(:))
    if ~isempty(shift_c_mex{ii})
        centerInd = segCenters(ii);
        centerIndRow = xI(centerInd) + dimb1(1);
        centerIndCol = yI(centerInd) + dimb1(2);
        centerIndHih = zI(centerInd) + dimb1(3);
        kSolveInds = col(linearIndsTile(centerIndRow:centerIndRow+segWidth-1,...
            centerIndCol:centerIndCol+segWidth-1,...
            centerIndHih:centerIndHih+segWidth-1));
        W(:,kSolveInds) = Wentries{ii};
    end
end
toc
