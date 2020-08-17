function W = kPtx(b1,k,offRes_args,k_pTx_args)


[dimb1(1),dimb1(2),dimb1(3),nCoils] = size(b1);

if isempty(offRes_args)
   ifOffRes=false;
else
    ifOffRes=true;
    b0=offRes_args{1};
    mask=offRes_args{2};
    dt=offRes_args{3};
    Lseg=offRes_args{4};
end


segWidth=k_pTx_args{1};
nHood=k_pTx_args{2};
Tik=k_pTx_args{3};
kWrapBack=k_pTx_args{4};
nThreads=k_pTx_args{5};

kposx = -dimb1(1)/2:dimb1(1)/2-1;
kposy = -dimb1(2)/2:dimb1(2)/2-1;
kposz = -dimb1(3)/2:dimb1(3)/2-1;
[kxPos,kyPos,kzPos]=meshgrid(kposx,kposy,kposz);  % in cycle/FOV

linearInds = reshape(1:dimb1(1)*dimb1(2)*dimb1(3),[dimb1(1) dimb1(2) dimb1(3)]);
[xI,yI,zI] = ndgrid(1:dimb1(1),1:dimb1(2),1:dimb1(3));

segCenters = linearInds(1:segWidth:end,1:segWidth:end,1:segWidth:end); % center indices of segments


kxTraj=k(:,1);
kyTraj=k(:,2);
kzTraj=k(:,3);


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
    kxTrajTile = kxTrajTile(:);kyTrajTile = kyTrajTile(:);kzTrajTile = kzTrajTile(:);
else

    kxTrajTile=kxTraj;
    kyTrajTile=kyTraj;
    kzTrajTile=kzTraj;

end


linearIndsTile = repmat(linearInds,[3 3 3 ]);

%%
if ifOffRes
        
    J = [6 6 6]; % # of neighnors used
    tb0 = (0:size(k,1)-1)*dt/1000-size(k,1)*dt/1000;
    b1_collapse = permute(b1,[4 1 2 3]);b1_collapse = b1_collapse(:,:).'; b1_collapse = b1_collapse(mask,:);
    nufft_args = {dimb1, J, 2*dimb1, dimb1/2, 'table', 2^10, 'minmax:kb'}; % NUFFT arguments
    % Important:
    % For design zmap should be +1i*2*pi*b0
    A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],mask,'fov',[1 1 1],'sens',conj(b1_collapse),'ti',tb0,'zmap',+1i*2*pi*b0,'L',Lseg,'nufft_args',nufft_args)';
    
    % apply off-resonance correction
    % embed spatial interpolators to sensitivity maps
    CtFull = zeros(Lseg,dimb1(1),dimb1(2),dimb1(3));
    for jj = 1:Lseg
    CtFull(jj,mask) = A.arg.Ct(:,jj);
    end
    CtFull = permute(CtFull,[2 3 4 1]);
    BFull=A.arg.B(1:end,:);

    BFull=single(BFull);
    BFull_c=cell(1);
    BFull_c{1}=BFull;
    
    
    %         calib=zeros(dimb1(1),dimb1(2),dimb1(3),nCoils,Lseg);
    %         for ii=1:Lseg
    %             calib(:,:,:,:,ii)=bsxfun(@times,sens,CtFull(:,:,:,ii));
    %         end

            calib=zeros(dimb1(1),dimb1(2),dimb1(3),nCoils,Lseg);
            for ii=1:Lseg
                for jj=1:nCoils
                calib(:,:,:,jj,ii)=b1(:,:,:,jj).*CtFull(:,:,:,ii);
                end
            end

            calib=reshape(calib,dimb1(1),dimb1(2),dimb1(3),nCoils*Lseg);
            nCoilsLseg=nCoils*Lseg;

    disp 'Building FT maps'
    tic
            nComb = nCoilsLseg*(nCoilsLseg+1)/2;
            calib = single(calib);
            FToversamp=1;   % Oversampling factor = 1 to save memory 
            [F_c, dgrid_StS] = FFTTrick(calib, nComb,FToversamp);

    toc
    
            [F_c_SensMapOnly, dgrid] = FFTTrick_SensMapOnly(calib, nComb,FToversamp);
    


else
    disp 'Building FT maps'
    tic
            nComb = nCoils*(nCoils+1)/2; % number of coil combinations
            calib = single(b1); % input B1 maps has to be single
            FToversamp=2;   % Oversampling factor of 
            [F_c, dgrid_StS] = FFTTrick(calib, nComb,FToversamp);
    toc
    
            [F_c_SensMapOnly, dgrid] = FFTTrick_SensMapOnly(calib, nComb,FToversamp);

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
      
      shift_c_mex{ii} =single(-[kyTrajTile(kTrajInds1)-kyPos(centerIndRow,centerIndCol,centerIndHih), kxTrajTile(kTrajInds1)-kxPos(centerIndRow,centerIndCol,centerIndHih), kzTrajTile(kTrajInds1)-kzPos(centerIndRow,centerIndCol,centerIndHih)])'; 
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
        [coeff_c]=LS_fft_mex_clean_OpenMP_Calloc(shift_c_mex, nCoils*Lseg, Tik, F_c, idgrid, shSolve', F_c_SensMapOnly,nCoils,BFull_c,kTrajInds_ii, nThreads);
    else
        [coeff_c]=LS_fft_mex_clean(shift_c_mex, nCoils*Lseg, Tik, F_c, idgrid, shSolve', F_c_SensMapOnly,nCoils,BFull_c,kTrajInds_ii);   
    end
else
    if  nThreads ~=0
        [coeff_c]=LS_fft_mex_clean_OpenMP_Calloc(shift_c_mex, nCoils, Tik, F_c, idgrid, shSolve', F_c_SensMapOnly, nThreads);
    else
        [coeff_c]=LS_fft_mex_clean(shift_c_mex, nCoils, Tik, F_c, idgrid, shSolve', F_c_SensMapOnly);
    end
end
toc

%%
disp 'Filling W matrix'
tic
clear Wentries 
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
        Wentries{ii} = sparse(i,j,double(coeff_c{ii}(:)),nCoils*length(kxTraj),segWidth^3);
        totalnonzero = totalnonzero + nCoils * length(kTrajInds_ii{ii}(:)) * segWidth^3;
        
    end
    
end

W = spalloc(length(kxTraj(:))*nCoils,dimb1(1)*dimb1(2)*dimb1(3),totalnonzero);
for ii = 1:length(segCenters(:))
    if ~isempty(shift_c_mex{ii})
        centerInd = segCenters(ii);
        centerIndRow = xI(centerInd)+dimb1(1);
        centerIndCol = yI(centerInd)+dimb1(2);
        centerIndHih = zI(centerInd)+dimb1(3);
        kSolveInds = col(linearIndsTile(centerIndRow:centerIndRow+segWidth-1,...
            centerIndCol:centerIndCol+segWidth-1,...
            centerIndHih:centerIndHih+segWidth-1));
        W(:,kSolveInds) = Wentries{ii};
    end
end
toc

end