addpath utils/
addpath utils/VERSE/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mls = true;            % switch to use magnitude least-squares

Tik = 3; %Regulerizer used in solving for W matrix.

flipAngle = 30;

genfigs = true;


b1MapSelect = '24loopsMartin';

kTrajSelect = 'spins';

dPatternSelect = 'midSelect';

ifOffRes=true;

ifOpenMP=true; % IN OpemMP, number of thread has to be change in the .c file and then re-compile

nThreads = 8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load b1 maps, k-space trajectory, target pattern
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch(b1MapSelect)       
        
    case '24loopsMartin'

        load('~/Dropbox/kspacePTX_data/MGH24loop_head1_128_2.mat')

        undersamp=4;  %Important parameter undersampling factor for b1 maps
        b1 = B1p3dxyz128;
        mask = logical(Mask3dxyz128);
        b1 = b1(1:undersamp:end,1:undersamp:end,1:undersamp:end,:);
        mask = mask(1:undersamp:end,1:undersamp:end,1:undersamp:end);
        
        fov = size(mask)*0.15*undersamp; % cm, res for full B1p3dxyz128 is 0.15 cm
        
        % If do off res correction, then build a B0 map
        if ifOffRes
            %%%% Important, only use 12 coils to save some computation time
            b1=b1(:,:,:,1:2:end);  
            %%%%    
            
            Lseg=4;  %Important parameter
            
            dim = size(B1p3dxyz128);  
            offResSigma = 3; offResCenterY = -3; % cm, width and center of Gaussian off-resonance field
            offResAmp = 500; % Hz
            [xb0,yb0,zb0] = ndgrid(-fov(1)/2:fov(1)/dim(1):fov(1)/2-fov(1)/dim(1),-fov(2)/2:fov(2)/dim(2):fov(2)/2-fov(2)/dim(2),-fov(3)/2:fov(3)/dim(3):fov(3)/2-fov(3)/dim(3));
            b0_128 = offResAmp * exp(-(xb0.^2 + (yb0-offResCenterY).^2 + zb0.^2)./offResSigma^2); % Hz

            b0 = b0_128(1:undersamp:end,1:undersamp:end,1:undersamp:end,:);
        else           
            b0 = zeros(size(mask));
        end
        
end



[dimb1(1),dimb1(2),dimb1(3),Nc] = size(b1);

% % normalize b1 by median so that regularization params can be
% % meaningfully reported
% b1Scale = 1/median(abs(b1(repmat(mask,[1 1 1 Nc]))));
% b1 = b1*b1Scale;
% % apply mask to b1 maps, if not already done
% %b1 = b1.*repmat(mask,[1 1 1 Nc]);



switch(kTrajSelect)
    case 'spins'
            k_accelerate=1;

            kmax = 0.75/2; % cycles/cm, max k-space loc
            T = 5; % ms, duration of pulse
            dt = 15e-3;
            %dt = 20e-3; % ms, dwell time
            t = 0:dt:T-dt;
            kr = linspace(kmax,0,length(t));
            u = 2*pi/(T/31);v = 2*pi/(T/17);% for kmax=1.5,dur 10ms
            u = 2*pi/(T/15.5/1);v = 2*pi/(T/8.5/1*4);
            u = u/k_accelerate;v = v/k_accelerate;
            ktheta = u*t;
            kphi = v*t;
            kx = kr.*cos(ktheta).*sin(kphi);
            ky = kr.*sin(ktheta).*sin(kphi);
            kz = kr.*cos(kphi);
            k = [kx(:) ky(:)  kz(:) ];
            Nt = size(k,1);
            NN = [Nt 0];
            
            T2=5;
            t2=0:dt:T2-dt;
            t2=-flip(t2);
            kr = linspace(1.25/2,0.75/2,length(t2));
            u = 2*pi/(T2/(31/3)/2);v = 2*pi/(T2/(17/3)*2);
            u = u/k_accelerate;v = v/k_accelerate;
            ktheta = u*t2;
            kphi = v*t2;
            kx2 = kr.*cos(ktheta).*sin(kphi);
            ky2 = kr.*sin(ktheta).*sin(kphi);
            kz2 = kr.*cos(kphi);
            k2 = [kx2(:) ky2(:)  kz2(:) ];
            k=cat(1,k2(1:end-1,:),k);            
            Nt = size(k,1);
            t = 0:dt:Nt*dt-dt;
            
            T3=5;
            t3=0:dt:T3-dt;
            t3=-flip(t3);
            kr = linspace(1,1.25/2,length(t3));
            u3 = 2*pi/(T3/15.5/1);v3 = 2*pi/(T3/8.5/1*4);
            u3 = u3/k_accelerate;v3 = v3/k_accelerate;
            ktheta = u3*t3+u*t2(1);
            kphi = v3*t3+v*t2(1);
            kx3 = kr.*cos(ktheta).*sin(kphi);
            ky3 = kr.*sin(ktheta).*sin(kphi);
            kz3 = kr.*cos(kphi);
            k3 = [kx3(:) ky3(:)  kz3(:) ];
            k=cat(1,k3(1:end-1,:),k);
            Nt = size(k,1);
            t = 0:dt:Nt*dt-dt;
            
            g=diff(k-k(1,:))/(1000*42.58*(dt/1000)/100); %mT/m
            g=g/10; %mT/m to G/cm
            smax = 700*100; %  %mT/m/ms to G/cm/s 
            gmax = 20; % g/cm

            %%%%% Parameter for conventional 7T
            %smax = 18000; %  %mT/m/ms to G/cm/s 
            %gmax = 4; % g/cm
           
            [b1v,gv] = toVERSE(zeros(size(g,1),1),g,dt/1000,100,smax,gmax,4257,1);
            k=cumsum(gv*10*(1000*42.58*(dt/1000)/100)); %1/cm
            k=k-k(end,:);

            t= 0:dt:length(k)*dt-dt;
            Nt= size(k,1);
            
            
            NN = [Nt 0];
            
end

switch dPatternSelect
    
        
    case 'midSelect'
        
      
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
             %d_smooth=d;
             d_128=d_smooth;
                         
             d = d_smooth(1:undersamp:end,1:undersamp:end,1:undersamp:end); 
             dim = size(d);
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


d = double(d);
pDes=fftshift(fftn(fftshift(d)));



            
%%
sens=b1;
sens=sens.*mask;
%sens=sens.*mask_ring;


nCoils=Nc;

% In this design, the FOVs and dimensions of the x and y dimensions are the same,
% but those of the y dimension are different.
FOV = fov(1); 
xyDim=dimb1(1);

FOVz = fov(3);
xyDimz = dimb1(3);

pos = -FOV/2:FOV/xyDim:FOV/2-FOV/xyDim;
posz = -FOVz/2:FOVz/xyDimz:FOVz/2-FOVz/xyDimz;
[xPos,yPos,zPos] = meshgrid(pos,pos,posz);

kpos = -xyDim/(2*FOV):1/FOV:xyDim/(2*FOV)-1/FOV;
kposz = -xyDimz/(2*FOVz):1/FOVz:xyDimz/(2*FOVz)-1/FOVz;
[kxPos,kyPos,kzPos]=meshgrid(kpos,kpos,kposz); 

%%

% process different sectors together - first just make sure we can do each one individually

linearInds = reshape(1:xyDim^2*xyDimz,[xyDim xyDim xyDimz]);
[xI,yI,zI] = ndgrid(1:xyDim,1:xyDim,1:xyDimz);
segWidth = 4;%4 width of segments in each direction
segCenters = linearInds(1:segWidth:end,1:segWidth:end,1:segWidth:end); % center indices of segments
kRadTraj = 4/FOV; %4
nHood = kRadTraj*FOV; % radius in terms of indices
nHoodz = 3;
%argCenters = reshape(1:length(kxPos(:)),[2 length(kxPos(:))/2]).';

genfig=false;

kxTraj=k(:,1);
kyTraj=k(:,2);
kzTraj=k(:,3);

% Whether excitation energy deposites across FOV to the other end through kspace wrap back. 
kWrapBack=false; 

if kWrapBack
    % tile the trajectory around the center to find traj points circulantly
    kxTrajTile = zeros(length(kxTraj(:)),27);
    kyTrajTile = zeros(length(kyTraj(:)),27);
    kzTrajTile = zeros(length(kzTraj(:)),27);

    for ii = 1:3
      for jj = 1:3
          for kk=1:3
        kxTrajTile(:,(ii-1)*3+jj+(kk-1)*9) = kxTraj + (ii-2)*xyDim/FOV;
        kyTrajTile(:,(ii-1)*3+jj+(kk-1)*9) = kyTraj + (jj-2)*xyDim/FOV;
        kzTrajTile(:,(ii-1)*3+jj+(kk-1)*9) = kzTraj + (kk-2)*xyDimz/FOVz;
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
    b1 = permute(sens,[4 1 2 3]);b1 = b1(:,:).'; b1 = b1(mask,:);
    nufft_args = {dimb1, J, 2*dimb1, dimb1/2, 'table', 2^10, 'minmax:kb'}; % NUFFT arguments
    % Important:
    % For design zmap should be +1i*2*pi*b0
    A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],mask,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1),'ti',tb0,'zmap',+1i*2*pi*b0,'L',Lseg,'nufft_args',nufft_args)';
    
    % apply off-resonance correction
    % embed spatial interpolators to sensitivity maps
    CtFull = zeros(Lseg,dimb1(1),dimb1(2),dimb1(3));
    for jj = 1:Lseg
    CtFull(jj,mask) = A.arg.Ct(:,jj);
    end
    CtFull = permute(CtFull,[2 3 4 1]);
    BFull=A.arg.B(NN(2)+1:end,:);

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
                calib(:,:,:,jj,ii)=sens(:,:,:,jj).*CtFull(:,:,:,ii);
                end
            end

            calib=reshape(calib,dimb1(1),dimb1(2),dimb1(3),nCoils*Lseg);
            nCoilsLseg=nCoils*Lseg;


    tic
            nComb = nCoilsLseg*(nCoilsLseg+1)/2;
            calib = single(calib);
            FToversamp=1;   % Oversampling factor = 1 to save memory 
            [F_c, dgrid_StS] = FFTTrick(calib, nComb,FToversamp);

    toc
    
            [F_c_SensMapOnly, dgrid] = FFTTrick_SensMapOnly(calib, nComb,FToversamp);
    


else
    tic
            nComb = nCoils*(nCoils+1)/2; % number of coil combinations
            calib = single(sens); % input B1 maps has to be single
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
tic

shift_c_mex=cell(1,length(segCenters(:)));
kTrajInds_ii=cell(1,length(segCenters(:)));

for ii = 1:length(segCenters(:)) 
     centerInd = segCenters(ii);
     centerIndRow = xI(centerInd);
     centerIndCol = yI(centerInd);
     centerIndHih = zI(centerInd);
 
 
     % find nearby points on trajectory, which can influence target segment

     kTrajInds = find(kxTrajTile*FOV >= kxPos(centerInd)*FOV-nHood & ...
                      kxTrajTile*FOV <= kxPos(centerInd)*FOV+segWidth-1+nHood & ...
                      kyTrajTile*FOV >= kyPos(centerInd)*FOV-nHood & ...
                      kyTrajTile*FOV <= kyPos(centerInd)*FOV+segWidth-1+nHood & ...
                      kzTrajTile*FOVz >= kzPos(centerInd)*FOVz-nHoodz & ...
                      kzTrajTile*FOVz <= kzPos(centerInd)*FOVz+segWidth-1+nHoodz);

     kTrajInds1=kTrajInds;
     kTrajInds_ii{ii} = single(mod(kTrajInds-1,length(kxTraj(:)))+1);
     %%% kTrajTile is for wrap back. and mod is to get rid of the index
     %%% increament (the sudden change) between wrap back.
      nNgb(ii)=length(kTrajInds1);
      
      shift_c_mex{ii} =single(-[kyTrajTile(kTrajInds1)-kyPos(centerIndRow,centerIndCol,centerIndHih), kxTrajTile(kTrajInds1)-kxPos(centerIndRow,centerIndCol,centerIndHih), kzTrajTile(kTrajInds1)-kzPos(centerIndRow,centerIndCol,centerIndHih)].*[FOV FOV FOVz])'; 
      % cycle/FOV, shift_c_mex is the relative shifts against the first
      % point of the target segment. It is needed for the interpolation.
      % shift_c_mex has to be cell of single matrices due to data handling of mex function.
end
toc



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
tic
if ifOffRes 
    if  ifOpenMP
        %[coeff_c]=LS_fft_mex_clean_OpenMP_SetMaxNhNSize_ForB0(shift_c_mex, nCoils*Lseg, Tik, F_c, idgrid, shSolve', F_c_SensMapOnly,nCoils,BFull_c,kTrajInds_ii);
        [coeff_c]=LS_fft_mex_clean_OpenMP_Calloc_ForB0(shift_c_mex, nCoils*Lseg, Tik, F_c, idgrid, shSolve', F_c_SensMapOnly,nCoils,BFull_c,kTrajInds_ii, nThreads);
    else
        [coeff_c]=LS_fft_mex_clean_ForB0(shift_c_mex, nCoils*Lseg, Tik, F_c, idgrid, shSolve', F_c_SensMapOnly,nCoils,BFull_c,kTrajInds_ii);   
    end
else
    if  ifOpenMP
        %[coeff_c]=LS_fft_mex_clean_OpenMP_SetMaxNhNSize(shift_c_mex, nCoils, Tik, F_c, idgrid, shSolve', F_c_SensMapOnly);
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

W = spalloc(length(kxTraj(:))*nCoils,xyDim^2*xyDimz,totalnonzero);
for ii = 1:length(segCenters(:))
    if ~isempty(shift_c_mex{ii})
        centerInd = segCenters(ii);
        centerIndRow = xI(centerInd)+xyDim;
        centerIndCol = yI(centerInd)+xyDim;
        centerIndHih = zI(centerInd)+xyDimz;
        kSolveInds = col(linearIndsTile(centerIndRow:centerIndRow+segWidth-1,...
            centerIndCol:centerIndCol+segWidth-1,...
            centerIndHih:centerIndHih+segWidth-1));
        W(:,kSolveInds) = Wentries{ii};
    end
end
toc

disp 'Getting RF'
tic
%rf = reshape(W*pDes(:)/numel(d_smooth),[length(kxTraj) nCoils]);
rf = reshape(W*pDes(:)/numel(d),[length(kxTraj) nCoils]);

toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
b1 = permute(sens,[4 1 2 3]);b1 = b1(:,:).';

 J = [6 6 6]; % # of neighnors used
 nufft_args = {dimb1, J, 2*dimb1, dimb1/2, 'table', 2^10, 'minmax:kb'}; % NUFFT arguments
 
 b1 = b1(mask,:);
 
 if ifOffRes
     A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],mask,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1),'ti',tb0,'zmap',-1i*2*pi*b0,'L',Lseg,'nufft_args',nufft_args)';
 else
     A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],mask,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1),'nufft_args',nufft_args)';
 end
    
 %%% the difference between ndgird and meshgrid makes here we need to shift
 %%% kx and ky.
 
 
m = zeros(dimb1);m(mask) = A*rf(:);
err = norm(col(mask.*(abs(m)-abs(d))))/norm(col(d.*mask))


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display final results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mask_ring=mask;
err = norm(col(mask_ring.*(abs(m)-abs(d))))/norm(col(d.*mask_ring))

d_plot=d.*mask_ring;
m_plot=m.*mask_ring;

if genfigs
    maxamp = max(abs([d_plot(:);m_plot(:)]));
    figure
    subplot(221)
    im(abs(d_plot),[0 maxamp]);axis image;colorbar
    title 'Desired pattern'
    subplot(222)
    im(abs(m),[0 maxamp]);axis image;colorbar
    caxis([0 1.1])
    title(sprintf('Final pattern',Nc));
    subplot(223)
    im(permute(abs(abs(m_plot)-abs(d_plot)),[1 3 2]));axis image;colorbar
    caxis([0 0.1])
    subplot(221)
    im(permute(abs(abs(m_plot)-abs(d_plot)),[2 3 1]));axis image;colorbar
    caxis([0 0.1])
    subplot(224)
    im(abs(abs(m_plot)-abs(d_plot)));axis image;colorbar
    caxis([0 0.1])
    title(sprintf('Error\nNRMSE = %0.2f%%',err*100));
end


% m_all(:,:,:,end+1)=m;
% end  %For slice by slice
% m_all(:,:,:,1)=[];



errmap=abs(abs(m)-abs(d.*mask));
map=zeros(192/undersamp,128/undersamp);
map(1:88/undersamp,1:128/undersamp)=squeeze(errmap((21-1)/undersamp+1:108/undersamp,:,48/undersamp));
map((89-1)/undersamp+1:end,1:64/undersamp)=squeeze(flip(errmap(64/undersamp,(21-1)/undersamp+1:124/undersamp,(21-1)/undersamp+1:84/undersamp),3));
map((97-1)/undersamp+1:184/undersamp,(65-1)/undersamp+1:end)=squeeze(flip(errmap((21-1)/undersamp+1:108/undersamp,72/undersamp,(21-1)/undersamp+1:84/undersamp),3));
figure
im(map,[0 0.1]);axis off;colorbar;title('');




%return
%%%%%%%%%%%%%
b1_128 = permute(B1p3dxyz128,[4 1 2 3]);b1_128 = b1_128(:,:).';
if ifOffRes
    b1_128 = permute(B1p3dxyz128(:,:,:,1:2:end),[4 1 2 3]);b1_128 = b1_128(:,:).';
end
b1_128 = b1_128(Mask3dxyz128,:);

dim=size(d_smooth);
J = [6 6 6]; % # of neighnors used
nufft_args = {dim, J, 2*dim, dim/2, 'table', 2^10, 'minmax:kb'}; % NUFFT arguments

        if max(abs(b0(:)) > 0)
            tb0 = (0:size(k,1)-1)*dt/1000-size(k,1)*dt/1000;
            Lseg = 8;
            A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],Mask3dxyz128,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1_128),'ti',tb0,'zmap',-1i*2*pi*b0_128,'L',Lseg,'nufft_args',nufft_args)';
        else
            A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],Mask3dxyz128,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1_128),'nufft_args',nufft_args)';
        end


%A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],Mask3dxyz128,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1_128),'nufft_args',nufft_args)';
m = zeros(size(Mask3dxyz128));m(Mask3dxyz128) = A*rf(:);
err = norm(col(Mask3dxyz128.*(abs(m)-abs(d_smooth))))/norm(col(d_smooth.*Mask3dxyz128))

errmap=abs(abs(m)-abs(d_smooth.*Mask3dxyz128));
%errmap=abs(abs(m));
map=zeros(192,128);
map(1:88,1:128)=squeeze(errmap(21:108,:,48));
map(89:end,1:64)=squeeze(flip(errmap(64,21:124,21:84),3));
map(97:184,65:end)=squeeze(flip(errmap(21:108,72,21:84),3));

figure
im(map,[0 0.1]);axis off;colorbar;title('');
%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate SAR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
if false
%if max(abs(b0(:)) > 0)
          b1 = permute(sens,[4 1 2 3]);b1 = b1(:,:).'; b1 = b1(mask,:);
          nufft_args_undersamp = {dimb1, J, 2*dimb1, dimb1/2, 'table', 2^10, 'minmax:kb'}; % NUFFT arguments
 
          A_undersamp = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],mask,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1),'ti',tb0,'zmap',-1i*2*pi*b0,'L',Lseg,'nufft_args',nufft_args_undersamp)';
          % apply off-resonance correction
          % embed spatial interpolators to full image matrix
          CtFull = zeros(Lseg,dimb1(1),dimb1(2),dimb1(3));
          for jj = 1:Lseg
            CtFull(jj,mask) = A_undersamp.arg.Ct(:,jj);
          end
          CtFull = permute(CtFull,[2 3 4 1]);
          BFull = repmat(A_undersamp.arg.B(NN(2)+1:end,:),[Nc 1]); % replicate to all coils
          rfk_offRes = 0;
          for jj = 1:Lseg
              rfk_offRes = rfk_offRes + BFull(:,jj).*(W*col(fftshift(fftn(fftshift(CtFull(:,:,:,jj).*d))))/numel(d));
          end
          mk_offRes = zeros(dimb1);mk_offRes(mask) = A_undersamp*rfk_offRes(:);
     
          err = norm(col(mask_ring.*(abs(mk_offRes)-abs(d))))/norm(col(d.*mask_ring))
end



%%
if false
rf_offRes_kspace_sum=zeros(length(kxTraj)*nCoils,1);
for ii=1:24
   ii_index=(ii-1)*32*32+1:ii*32*32;
   A_idft=exp(-1i*2*pi.*(kxPos(:)*xPos(ii_index)+kyPos(:)*yPos(ii_index)+kzPos(:)*zPos(ii_index)));
   rf_offRes_kspace=W*(A_idft.*d(ii_index))/numel(d);
   rf_offRes_kspace=rf_offRes_kspace.*repmat(exp(1i*2*pi.*tb0(:)*b0(ii_index)),24,1);
   rf_offRes_kspace_sum=sum(rf_offRes_kspace,2)+rf_offRes_kspace_sum;
   ii
end
    
rf_offRes_kspace_sum = reshape(rf_offRes_kspace_sum,[length(kxTraj) nCoils]);
end

%%
if false
 b1 = permute(sens,[4 1 2 3]);
%[xPos,yPos] = ndgrid(pos); xPos made before is using meshgrid
 A_exact = exp(1i*2*pi.*(xPos(:)*kxTraj(:)'+yPos(:)*kyTraj(:)'+zPos(:)*kzTraj(:)'));
 if max(abs(b0(:)) > 0)
 A_exact = exp(1i*2*pi.*(xPos(:)*kxTraj(:)'+yPos(:)*kyTraj(:)'+zPos(:)*kzTraj(:)'-b0(:)*tb0(:)'));
 end
 m = zeros(xyDim,xyDim,xyDimz,nCoils);
for ii = 1:nCoils
    m(:,:,:,ii) = squeeze(b1(ii,:,:,:)).*reshape(A_exact*rf(:,ii),[xyDim xyDim xyDimz]);
end
%nrmse(Pdes,sum(m,4),1);
m=sum(m,4);
figure
im(m)

m = m.*mask_ring;
err = norm(col(mask_ring.*(abs(m)-abs(d))))/norm(col(d.*mask_ring))
end