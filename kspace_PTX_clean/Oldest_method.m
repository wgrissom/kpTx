% same as delta_wts_test, but loop over each target point and build
% individual kernel, then try reduced sampling pattern. also adding spiral
% traj

% next steps:
% 2) parallelize!
% 3) test off resonance
% 4) test 3D
% 5) Large-tip

FOV = 20;
%xyDim = 32;

traj = 'spiral'; % 'spiral' or 'cartesian'
accfactor = 3;
dPattern = 'circle'; % 'square' or 'circle'
kRadTraj = 4/FOV; % in fitting weights, consider radius of 5 k-locations around target point

genfig = true;

% load sensitivities
disp 'loading sensitivities'
%load ~/data/data_FROMOLDDRIVE/xsense_sim/sensitivities/sensitivities_0503_1_masked
load('sensitivities_0503_1_masked.mat')
nCoils = size(sens32,1);
for ii = 1:nCoils
    sens32(ii,:,:) = squeeze(sens32(ii,:,:)).*bodymask32;
end
%sens32 = sens32(:,4:27,3:26);
xyDim = size(sens32,2);

% get k-locations
disp 'defining trajectories'

switch traj
    case 'cartesian' % uniform sampling
        kpos = -xyDim/(2*FOV):1/FOV:xyDim/(2*FOV)-1/FOV;
        [kxPos,kyPos]=meshgrid(kpos); % don't need this if doing traj
%%%%% Martin %%%%%% I don't think you have shift them by falf a k-voxel??
% shift them by half a k-voxel,
%%%%% Martin %%%%%% I don't think you have truncated anything?
% truncate to neighborhood around DC (which is the target point)
        kxTraj = kxPos(1:accfactor:end,:); % shift grid by half a sampling point,
        kyTraj = kyPos(1:accfactor:end,:); % relative to original grid
    case 'spiral' % constant angular rate spiral
        kpos = -xyDim/(2*FOV):1/FOV:xyDim/(2*FOV)-1/FOV;
        [kxPos,kyPos]=meshgrid(kpos);
        nK = xyDim^2/accfactor; % # points on trajectory
        t = (0:nK-1)/nK;
        kmax = xyDim/(2*FOV); % /cm, max radius in k-space
        nTurns = kmax/(1/FOV)/accfactor; % number of turns
        k = kmax*(1-t).*exp(1i*2*pi*nTurns*(1-t)); % trajectory, /cm - kx = real(k); ky = imag(k);
        kxTraj = real(k);
        kyTraj = -imag(k);
end
% kxTraj,kyTraj time FOV will be in cycle/FOV, which is the "kTraj" used in grappaTestScript



pos = -FOV/2:FOV/xyDim:FOV/2-FOV/xyDim;
[xPos,yPos] = meshgrid(pos);

% now use weights to project sens-weighted target pattern onto entire
% trajectory
switch dPattern
    case 'circle'
        Pdes = sqrt(xPos.^2+yPos.^2)<=3;
    case 'square'
        Pdes = abs(xPos) <= 3 & abs(yPos) <= 3;
    case 'const'
        Pdes = ones(xyDim);
end
Pdes = double(Pdes);
%%% Image domian desired pattern

% go to Fourier domain
%pDes = ft2(Pdes);
pDes = fftshift(fft2(fftshift(Pdes)));
%%% kspace domain desired pattern.

% filter target pattern to 1% at max trajectory k-loc



pDes = pDes.*fermi(xyDim,24/32*12,1);
%%% Get rid of Gibbs ringing

%Pdes = ift2(pDes);
Pdes = fftshift(ifft2(fftshift(pDes)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%   What is this ?? %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% precalculate matrix of DFT'd sensitivities
% first, truncate the sensitivities' by taking FT, truncating, and going
% back to image domain
% (skip this part at first actually - will be less accurate than current method
% and won't save much here due to large neighborhood versus matrix size)
% second, shift each truncated sensitivity to each traj location
% third, apply DFT matrix to get shifted sensitivities in k-domain
% that completes the pre-processing
tic
% build a DFT matrix to take maps into k-domain
[xI,yI] = ndgrid(-xyDim/2:xyDim/2-1);
xItrunc = xI(abs(xI) > kRadTraj*FOV | abs(yI) > kRadTraj*FOV);
yItrunc = yI(abs(xI) > kRadTraj*FOV | abs(yI) > kRadTraj*FOV);
dftMtx = exp(-1i*2*pi/xyDim*(xItrunc(:)*xI(:)' + yItrunc(:)*yI(:)'));
mask = squeeze(sens32(1,:,:)) ~= 0;
sensProc = sens32;
dftMtxOut = dftMtx(:,~mask(:));
%lam = 0;
lam = 0.000001*norm(dftMtxOut'*dftMtxOut,'fro');
for ii = 1:nCoils
    sigIn = dftMtx*col(squeeze(sensProc(ii,:,:)));
    %sigOut = -dftMtxOut\sigIn;
    sigOut = -(dftMtxOut'*dftMtxOut + lam*eye(size(dftMtxOut'*dftMtxOut,1)))\(dftMtxOut'*sigIn);
    tmp = squeeze(sensProc(ii,:,:));%
    tmp(~mask) = sigOut;
    sensProc(ii,:,:) = tmp;
end
sens32Orig = sens32;
%sens32 = sensProc;


% Fist of all kRadTraj=4/FOV, which means in fitting weights, consider radius of 5 k-locations around target point
% It was supposed to be used only in fitting. But we also adapt this 5 k-location regions here

% xItrunc and yItrunc is the k location out side of a 5 k-location of DC (which is everything except for the center 9x9)
% So dftMtx is a dft matrix that takes the B1 map into k domain, but only caring the outside region of kspace

% dftMxtOut is a dft matrix connecting the no-brain region of the image domain B1 map and the outside region of the kspace domian B1 map
% sigOut is a least square erro solution of the no-brain region of the image domain B1 map, solved from the actual ouside region of the kspace domain B1 map

% The reason of doing this is we can only measure the B1 map inside the brain
% But we would want to know everything inside the FOV


% get the trajectory shift matrix
Ashift = exp(1i*2*pi*(xPos(:)*kxTraj(:)' + yPos(:)*kyTraj(:)'));
%%% xPos is in cm, kxTraj is in 1/cm.
%%% So this is time shift in Fourier transform instead of in DFT
%%% Since later sense.*xPos, with respect to each kxTraj, 
%%% it is shift acros each kxTraj
%%% Ashift is only the phase ramp in image domain

sensShift = zeros([length(xPos(:)) length(kxTraj) nCoils]);
for ii = 1:nCoils
    sensShift(:,:,ii) = bsxfun(@times,col(squeeze(sens32(ii,:,:))),Ashift);
end
dftMtx = exp(-1i*2*pi/xyDim*(xI(:)*xI(:)' + yI(:)*yI(:)'));
sensShiftDFT = reshape(dftMtx * sensShift(:,:),[xyDim xyDim length(kxTraj) nCoils]);
% sensShiftDFT is the k domain sense shifted according to different kTraj
% point.

%%

% then, loop over target locations and build S matrices that can proceed
% just as before.

% then, look at preprocessing and then truncating the sensitivities to 4*radius matrix dim,
% shifting them by a fractional amount,
% and then using relative look-up indices

% loop through target k-locs, projecting energy onto trajectory
nTraj = zeros(length(kxPos(:)),1);
nTarg = zeros(length(kxPos(:)),1);
Wentries = {};
%kTrajIndsAllEntries = {};

% process different sectors together - first just make sure we can do each one individually
%segCenters = (1:length(kxPos(:)))'; % each row is one group
linearInds = reshape(1:xyDim^2,[xyDim xyDim]);
[xI,yI] = ndgrid(1:xyDim);
segWidth = 4;% width of segments in each direction
segCenters = linearInds(1:segWidth:end,1:segWidth:end); % center indices of segments
nHood = kRadTraj*FOV; % radius in terms of indices
%argCenters = reshape(1:length(kxPos(:)),[2 length(kxPos(:))/2]).';

% tile the trajectory around the center to find traj points circulantly
kxTrajTile = zeros(length(kxTraj(:)),9);
kyTrajTile = zeros(length(kyTraj(:)),9);
for ii = 1:3
  for jj = 1:3
    kxTrajTile(:,(ii-1)*3+jj) = kxTraj + (ii-2)*xyDim/FOV;
    kyTrajTile(:,(ii-1)*3+jj) = kyTraj + (jj-2)*xyDim/FOV;
  end
end
kxTrajTile = kxTrajTile(:);kyTrajTile = kyTrajTile(:);

%kxTrajTile=kxTraj;
%kyTrajTile=kyTraj;

%figure
%plot(kxTrajTile,kyTrajTile)

linearIndsTile = repmat(linearInds,[3 3]);

% All the Tiling is for wrap back in shifting

for ii = 1:length(segCenters(:)) % loop over sectors

    centerInd = segCenters(ii);
    centerIndRow = xI(centerInd);
    centerIndCol = yI(centerInd);

    kSolveInds = linearInds(centerIndRow:centerIndRow+segWidth-1,...
                            centerIndCol:centerIndCol+segWidth-1);
    kSolveIndsVec = kSolveInds(:);

    % find nearby points on trajectory, which can influence target point
    %kTrajInds = find(sqrt((kxTraj-kxPos(ii)).^2 + (kyTraj-kyPos(ii)).^2) <= kRadTraj);
    % find all trajectory points that influence this segment
    % base this on middle point of segment
    kTrajInds = find(kxTrajTile*FOV >= kxPos(centerInd)*FOV-nHood & ...
                     kxTrajTile*FOV <= kxPos(centerInd)*FOV+segWidth-1+nHood & ...
                     kyTrajTile*FOV >= kyPos(centerInd)*FOV-nHood & ...
                     kyTrajTile*FOV <= kyPos(centerInd)*FOV+segWidth-1+nHood);
    kTrajInds = mod(kTrajInds-1,length(kxTraj(:)))+1;
    %%% kTrajTile is for wrap back. and mod is to get rid of the index
    %%% increament (the sudden change) between wrap back.
    
    %kTrajInds = find(abs(kxTraj-(kxPos(centerInd)+(segWidth-1)/2/FOV)) <= kRadTraj+(segWidth-1)/2/FOV & ...
    %    abs(kyTraj-(kyPos(centerInd)+(segWidth-1)/2/FOV)) <= kRadTraj+(segWidth-1)/2/FOV);
    %%% The method above cannot handle wrap back
    
    %kTrajInds = find(abs(kxTraj-kxPos(centerInd)) <= kRadTraj & ...
    %    abs(kyTraj-kyPos(centerInd)) <= kRadTraj);
    %%% The method above did not have the extension for point center+segWidth
    
    %kxTrajTile is in 1/FOV. kxPos is also in 1/FOV.
    %kxTrajTile*FOV and kxPos*FOV are in there k space index unit.
    
    nTraj(ii) = length(kTrajInds);

    if ~isempty(kTrajInds)

        % find other target points near the trajectory points, which would be
        % influenced by the RF on those points. Could also just double the
        % original radius of the trajectory inds, which would be larger but
        % simpler to find
        %kTargInds = find(sqrt((kxPos-kxPos(ii)).^2 + (kyPos-kyPos(ii)).^2) <= 2*kRadTraj);

        kTargInds = linearIndsTile(centerIndRow-2*nHood+xyDim:...
                                   centerIndRow+segWidth-1+2*nHood+xyDim,...
                                   centerIndCol-2*nHood+xyDim:...
                                   centerIndCol+segWidth-1+2*nHood+xyDim);
        kTargIndsVec = kTargInds(:);
        if genfig
          figure(1);clf;hold on
          h = plot(kxPos(:)+1i*kyPos(:),'.');
          set(h,'Color',[0 0 0]);
          %h = plot(kxPos(kTargIndsVec)+1i*kyPos(kTargIndsVec),'o');
          %set(h,'MarkerFaceColor', get(h,'Color')) % Red
          %set(h,'MarkerSize', 4)
          h = plot(kxPos(kSolveIndsVec)+1i*kyPos(kSolveIndsVec),'o');
          yellowpoint=get(h,'Color');
          set(h,'MarkerFaceColor', get(h,'Color')) % Yello
          h = plot(kxTraj+1i*kyTraj,'-');
          set(h,'Color', [0 0 0]);
          h = plot(kxTraj(kTrajInds)+1i*kyTraj(kTrajInds),'o');
          set(h,'MarkerFaceColor', get(h,'Color'))  % Green
          set(h,'MarkerFaceColor', 'g')
          set(h,'MarkerSize', 5)
          h = plot(kxPos(kSolveIndsVec)+1i*kyPos(kSolveIndsVec),'o');
          set(h,'Color', yellowpoint) % Yello
          set(h,'MarkerFaceColor', yellowpoint) % Yello
          set(h,'MarkerSize', 7)
          axis([-xyDim/(2*FOV) xyDim/(2*FOV) -xyDim/(2*FOV) xyDim/(2*FOV)]);axis equal
          axis off
          drawnow;pause(0.5)
          pause;
        end
        %if ii == 6;return;end
        %kTargInds = find(abs(kxPos-(kxPos(centerInd)+(segWidth-1)/2)) <= 2.01*kRadTraj+(segWidth-1)/2 & ...
        %     abs(kyPos-(kyPos(centerInd)+(segWidth-1)/2)) <= 2.01*kRadTraj+(segWidth-1)/2);
        nTarg(ii) = length(kTargInds);

        S = zeros(length(kTargIndsVec),length(kTrajInds),nCoils);
        Srhs = zeros(segWidth*segWidth,length(kTrajInds),nCoils);

        % build the S matrix including these points
        for jj = 1:length(kTrajInds)

            % grab the sensitivity entries we need
            for kk = 1:nCoils
                tmp = sensShiftDFT(:,:,kTrajInds(jj),kk);
                S(:,jj,kk) = tmp(kTargIndsVec);
                Srhs(:,jj,kk) = tmp(kSolveIndsVec);
            end

        end

        % collapse coil and trajectory k-locs dims
        Scollapse = S(:,:);
        SrhsCollapse = Srhs(:,:);

        % solve for weights, with regularization
        StS = Scollapse'*Scollapse;
        lam = 0.0001*norm(StS,'fro');
        % which indices of kTargInds are the points we are solving for?
        %midInd = (kxPos(kTargIndsVec)-kxPos(centerInd) == 0) & ...
        %    (kyPos(kTargIndsVec)-kyPos(centerInd) == 0);
        w = (StS + lam*eye(size(StS,1)))\SrhsCollapse';%Scollapse(midInd,:)';

        % store the coefficients in W matrix
        kTrajIndsAll = repmat(kTrajInds(:),[1 nCoils]);
        for ll = 2:nCoils
            kTrajIndsAll(:,ll) = kTrajIndsAll(:,ll) + (ll-1)*length(kxTraj);
        end

        %Wentries{ii} = w;
        %kTrajIndsAllEntries{ii} = kTrajIndsAll(:);
        %%% The big W matrix has nCoils*length(kxTraj) rows
        %%% However, w is much less rows than W.
        %%% Here we only considering nTaj(ii) rows among W.
        %%% The rest of the row is assumed to be zero since they are not
        %%% influencing these 16 k locations. (W is sparse)
        
        %%% Without implementing Wentries sparse, will make building the
        %%% big W harder.
        
        [i,j] = ndgrid(kTrajIndsAll(:),1:segWidth^2);
        Wentries{ii} = sparse(i,j,w(:),nCoils*length(kxTraj),segWidth^2);
        %%% We made a sparse matrix whose size is nCoils*length(kxTraj)by segWidth^2
        %%% We full the (i(k),j(k)) element with w(k)
        %%% The kTrajIndsAll is to take care of the row index increament
        %%% across different coils
        
        %Wentries{centerInd} = w;
        %kTrajIndsAllEntries{centerInd} = kTrajIndsAll(:);
        %W(kTrajIndsAllEntries{ii},ii) = Wentries{ii};

    else

      %for ll = 1:segWidth
      %  for mm = 1:segWidth
      %    Wentries{kSolveInds(ll,mm)} = [];
      %    kTrajIndsAllEntries{kSolveInds(ll,mm)} = [];
      %  end
      %end
        %kTrajIndsAllEntries{ii} = [];
        Wentries{ii} = spalloc(nCoils*length(kxTraj),segWidth^2,0);
        %%% Allocate space for sparse matrix
    end

end
toc

tic
% stick the w entries into the big W matrix
W = spalloc(length(kxTraj(:))*nCoils,xyDim^2,0);
for ii = 1:length(segCenters(:))
  centerInd = segCenters(ii);
  centerIndRow = xI(centerInd);
  centerIndCol = yI(centerInd);
  kSolveInds = col(linearInds(centerIndRow:centerIndRow+segWidth-1,...
                          centerIndCol:centerIndCol+segWidth-1));

  W(:,kSolveInds) = Wentries{ii};
end


% need to fix this? would be best to have stacked indices and entries, and
% use sparse() function to build W
%tic
%for ii = 1:length(segCenters(:))
%  if ~isempty(Wentries{ii})
%    centerInd = segCenters(ii);
%    centerIndRow = xI(centerInd);
%    centerIndCol = yI(centerInd);
%
%    kSolveInds = linearInds(centerIndRow:centerIndRow+segWidth-1,...
%                            centerIndCol:centerIndCol+segWidth-1);
%    for ll = 1:segWidth
%      for mm = 1:segWidth
%        W(kTrajIndsAllEntries{ii},kSolveInds(ll,mm)) = Wentries{ii}(:,(mm-1)*segWidth+ll);
%      end
%    end
%  end
%end
%for ii = 1:length(kxPos(:))
%    if ~isempty(kTrajIndsAllEntries{ii})
%        W(kTrajIndsAllEntries{ii},ii) = Wentries{ii};
%    end
%end

% design pulses
rf = reshape(W*pDes(:),[length(kxTraj) nCoils]);
toc

%rf2 = kSpacepTxDes2(sens32,FOV,kxTraj,kyTraj,Pdes,segWidth,0.00001);

%pDesTest = reshape(W\rf(:),[xyDim xyDim]);
%nrmse(Pdes,ift2(pDesTest),1)

% evaluate pulses
A = exp(1i*2*pi.*(xPos(:)*kxTraj(:)'+yPos(:)*kyTraj(:)'));
m = zeros(xyDim,xyDim,nCoils);
for ii = 1:nCoils
    m(:,:,ii) = squeeze(sens32Orig(ii,:,:)).*reshape(A*rf(:,ii),[xyDim xyDim]);
end

nrmse(Pdes,sum(m,3),1)
norm(rf)

% compare to spatial-domain method
tic
A = exp(1i*2*pi.*(xPos(:)*kxTraj(:)'+yPos(:)*kyTraj(:)'));
sens32_collapse = sens32Orig(:,:).';
Afull = repmat(A,[1 1 nCoils]);
for ii = 1:nCoils
    Afull(:,:,ii) = bsxfun(@times,sens32_collapse(:,ii),Afull(:,:,ii));
end
Afull = Afull(:,:);
lam = 0.00001*norm(Afull,'fro');
rf_spatialDomain = reshape((Afull'*Afull + lam*eye(size(Afull,2)))\(Afull'*Pdes(:)),[length(kxTraj) nCoils]);
toc
m_spatialDomain = reshape(Afull*rf_spatialDomain(:),[xyDim xyDim]);

nrmse(Pdes,m_spatialDomain,mask)
norm(rf_spatialDomain)
