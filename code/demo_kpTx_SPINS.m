addpath utils/
addpath utils/VERSE/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mls = true;            % switch to use magnitude least-squares

flipAngle = 90;

genfigs = true;

if_kpsce_MLS = false;

ifOffRes = false;

ifOpenMP = true; % IN OpemMP, number of thread has to be change in the .c file and then re-compile

ifDoSpatialDesign = true;

if ifOpenMP
    if ~exist('kpTx_solve_w_OpenMP')
        mex -largeArrayDims -lmwlapack CXXFLAGS="$CXXFLAGS -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" COPTIMFLAGS="$COPTIMFLAGS -fopenmp -O2" LDOPTIMFLAGS="$LDOPTIMFLAGS -fopenmp -O2" DEFINES="$DEFINES -fopenmp" kpTx_solve_w_OpenMP.c
    end
    nThreads = 8;
else
    if ~exist('kpTx_solve_w')
        mex -largeArrayDims -lmwlapack kpTx_solve_w.c
    end
    nThreads = 0;
end

%%%% Uncomment only if want OpenMP

if ~exist('blochsim_optcont_mex')
    cd utils/
    mex -largeArrayDims -lmwlapack blochsim_optcont_mex.c
    cd ../
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load b1 maps, k-space trajectory, target pattern
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load('~/Dropbox/kspacePTX_data/MGH24loop_head1_128_2.mat')
load('MGH24loop_head1_128_2.mat')
load('Mask3dxyz128_ring.mat')

undersamp=2;  %Important parameter undersampling factor for b1 maps
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
    
    dim = size(B1p3dxyz128);
    offResSigma = 3; offResCenterY = -3; % cm, width and center of Gaussian off-resonance field
    offResAmp = 200; % Hz
    [xb0,yb0,zb0] = ndgrid(-fov(1)/2:fov(1)/dim(1):fov(1)/2-fov(1)/dim(1),-fov(2)/2:fov(2)/dim(2):fov(2)/2-fov(2)/dim(2),-fov(3)/2:fov(3)/dim(3):fov(3)/2-fov(3)/dim(3));
    b0_128 = offResAmp * exp(-(xb0.^2 + (yb0-offResCenterY).^2 + zb0.^2)./offResSigma^2); % Hz
    
    b0 = b0_128(1:undersamp:end,1:undersamp:end,1:undersamp:end,:);
    
else
    b0 = zeros(size(mask));
end


[dimb1(1),dimb1(2),dimb1(3),Nc] = size(b1);

%% Design SPINS Trajectory
k_accelerate=1;

kmax = 0.75/2; % cycles/cm, max k-space loc
T = 5; % ms, duration of pulse
dt = 15e-3;
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

g=-flip(diff([zeros(1,3);flip(k)],1,1))/(1000*42.58*(dt/1000)/100);%mT/m
g=g/10; %mT/m to G/cm
smax = 700*100; %  %mT/m/ms to G/cm/s
gmax = 20; % g/cm

[b1v, gv] = toVERSE(zeros(size(g,1),1), g, dt/1000, 100, smax, gmax, 4257, 1);
k = -flip(cumsum(flip(gv*10*(1000*42.58*(dt/1000)/100))));

t= 0:dt:length(k)*dt-dt;
Nt= size(k,1);


NN = [Nt 0];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d_ori=d;
%d_smooth=d;
d_128=d_smooth;

d = d_smooth(1:undersamp:end,1:undersamp:end,1:undersamp:end);
dim = size(d);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% take the FT of the desired pattern for kpTx design
d = double(d);
pDes=fftshift(fftn(fftshift(d)));

%%
sens=b1;
sens=sens.*mask;

segWidth = 4;%4 width of segments in each direction
nHood = 4; % radius in terms of indices
Tik = 3; %Regularizer used in solving for W matrix.
% Whether excitation energy deposites across FOV to the other end through kspace wrap back.
kWrapBack = false;

k_pTx_args={segWidth,nHood,Tik,kWrapBack,nThreads};


if ifOffRes
    Lseg = 4;  % Number of time segments in off-resonance model
    offRes_args = {b0, mask, dt, Lseg};
else
    offRes_args = [];
end

% k trajactory in cycle/FOV
W = kpTx(sens, k.*fov, k_pTx_args, offRes_args);


%%

disp 'Getting RF'
tic
rf = reshape(W*pDes(:)/numel(d), [size(k, 1) Nc]);
toc


%%
b1 = permute(sens,[4 1 2 3]);b1 = b1(:,:).';

J = [6 6 6]; % # of neighnors used
nufft_args = {dimb1, J, 2*dimb1, dimb1/2, 'table', 2^10, 'minmax:kb'}; % NUFFT arguments

b1 = b1(mask,:);

if ifOffRes
    tb0 = (0:size(k,1)-1)*dt/1000-size(k,1)*dt/1000;
    A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],mask,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1),'ti',tb0,'zmap',-1i*2*pi*b0,'L',Lseg,'nufft_args',nufft_args)';
else
    A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],mask,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1),'nufft_args',nufft_args)';
end

%%% the difference between ndgird and meshgrid makes here we need to shift
%%% kx and ky.


m = zeros(dimb1);m(mask) = A*rf(:);
err = norm(col(mask.*(abs(m)-abs(d))))/norm(col(d.*mask))

if if_kpsce_MLS
    m2=m;
    for ii=1:3
        pDes2=fftshift(fftn(fftshift(abs(d).*exp(1i*angle(m2)))));
        rf2 = reshape(W*pDes2(:)/numel(d),[size(k,1) Nc]);
        m2 = zeros(dimb1);m2(mask) = A*rf2(:);
        err = norm(col(mask.*(abs(m2)-abs(d))))/norm(col(d.*mask))
    end
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display final results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mask_ring=mask;
%err = norm(col(mask_ring.*(abs(m)-abs(d))))/norm(col(d.*mask_ring))

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



%%%%%%%%%%%%% Gmri simulation with 128x128x96 matrix (highest res)

% b1_128 = permute(B1p3dxyz128,[4 1 2 3]);b1_128 = b1_128(:,:).';
% if ifOffRes
%     b1_128 = permute(B1p3dxyz128(:,:,:,1:2:end),[4 1 2 3]);b1_128 = b1_128(:,:).';
% end
% b1_128 = b1_128(Mask3dxyz128,:);
%
% dim=size(d_smooth);
% J = [6 6 6]; % # of neighnors used
% nufft_args = {dim, J, 2*dim, dim/2, 'table', 2^10, 'minmax:kb'}; % NUFFT arguments
%
%         if max(abs(b0(:)) > 0)
%             tb0 = (0:size(k,1)-1)*dt/1000-size(k,1)*dt/1000;
%             Lseg = 8;
%             A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],Mask3dxyz128,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1_128),'ti',tb0,'zmap',-1i*2*pi*b0_128,'L',Lseg,'nufft_args',nufft_args)';
%         else
%             A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],Mask3dxyz128,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1_128),'nufft_args',nufft_args)';
%         end
%
%
% %A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],Mask3dxyz128,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1_128),'nufft_args',nufft_args)';
% m = zeros(size(Mask3dxyz128));m(Mask3dxyz128) = A*rf(:);
% err = norm(col(Mask3dxyz128.*(abs(m)-abs(d_smooth))))/norm(col(d_smooth.*Mask3dxyz128))
%
% errmap=abs(abs(m)-abs(d_smooth.*Mask3dxyz128));
% %errmap=abs(abs(m));
% map=zeros(192,128);
% map(1:88,1:128)=squeeze(errmap(21:108,:,48));
% map(89:end,1:64)=squeeze(flip(errmap(64,21:124,21:84),3));
% map(97:184,65:end)=squeeze(flip(errmap(21:108,72,21:84),3));
%
% figure
% im(map,[0 0.1]);axis off;colorbar;title('');

%%%%%%%%%%%%


%%%% Bloch_sim simulation

dim=size(d_smooth);
posx = -fov(1)/2:fov(1)/dim(1):fov(1)/2-fov(1)/dim(1);
posy = -fov(2)/2:fov(2)/dim(2):fov(2)/2-fov(2)/dim(2);
posz = -fov(3)/2:fov(3)/dim(3):fov(3)/2-fov(3)/dim(3);
[xPos,yPos,zPos] = meshgrid(posx,posy,posz);
xx=[xPos(Mask3dxyz128),yPos(Mask3dxyz128),zPos(Mask3dxyz128)];
sensBloch=B1p3dxyz128;
if ifOffRes
    sensBloch=sensBloch(:,:,:,1:2:end);
end
sensBloch = reshape(complex(sensBloch(repmat(Mask3dxyz128,1,1,1,size(sensBloch,4)))),[],size(sensBloch,4));
garea=-flip(diff([zeros(1,3);flip(k)],1,1))*2*pi;
garea=gv*10*(1000*42.58*(dt/1000)/100)*2*pi;
omdt=0*xPos(Mask3dxyz128);
if ifOffRes
    omdt=b0_128(Mask3dxyz128)*dt/1000*2*pi;
end
[a, b]=blochsim_optcont_mex(ones(Nt,1),ones(Nt,1),rf*flipAngle/180*pi,sensBloch,garea,xx,omdt,1); %rf*dt*4258*2*pi

Mxy=embed(2*conj(a).*b,Mask3dxyz128);
Mz=embed(1-2*abs(b).^2,Mask3dxyz128);

Mxy_RMSE = sqrt(mean((abs(Mxy(Mask3dxyz128_ring))-abs(d_smooth(Mask3dxyz128_ring))).^2))

Mz_RMSE = sqrt(mean((abs(Mz(Mask3dxyz128_ring))-abs(d_ori(Mask3dxyz128_ring)-1)).^2))

errmap=abs(abs(Mxy)-abs(d_smooth)).*Mask3dxyz128_ring;
map=zeros(192,128);
map(1:88,1:128)=squeeze(errmap(21:108,:,48));
map(89:end,1:64)=squeeze(flip(errmap(64,21:124,21:84),3));
map(97:184,65:end)=squeeze(flip(errmap(21:108,72,21:84),3));

figure
im(map,[0 0.1]);axis off;colorbar;title('Mxy Error');


%% Spatial
if ifDoSpatialDesign
    
    b1 = permute(sens,[4 1 2 3]);b1 = b1(:,:).';
    
    J = [6 6 6]; % # of neighnors used
    nufft_args = {dimb1, J, 2*dimb1, dimb1/2, 'table', 2^10, 'minmax:kb'}; % NUFFT arguments
    
    b1 = b1(mask,:);
    
    if ifOffRes
        tb0 = (0:size(k,1)-1)*dt/1000-size(k,1)*dt/1000;
        A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],mask,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1),'ti',tb0,'zmap',-1i*2*pi*b0,'L',Lseg,'nufft_args',nufft_args)';
    else
        A = Gmri_SENSE([k(:,2) k(:,1) k(:,3)],mask,'fov',[fov(2) fov(1) fov(3)],'sens',conj(b1),'nufft_args',nufft_args)';
    end
    
    ncgiters = 35;
    mls=false;
    disp 'Spatial Design'
    tic
    if ~mls
        rf = qpwls_pcg(zeros(length(k)*Nc,1),A,1,d(mask),0,1,1,ncgiters,mask); % CG
    else
        rf = qpwls_pcg(zeros(length(k)*Nc,1),A,1,d(mask),0,1,1,5,mask); % CG
        rfOld = 10*rf(:,end);
        %while norm(rfOld-rf(:,end)) > 0.01*norm(rfOld)
        for kk=1:5
            % update target phase if magnitude least-squares
            m = A*rf(:,end);
            d(mask) = abs(d(mask)).*exp(1i*angle(m));
            err = norm(col((abs(m)-abs(d(mask)))))/norm(col(d(mask)))
            rfOld = rf(:,end);
            rf = qpwls_pcg(rf(:,end),A,1,d(mask),0,Rfull,1,5,mask); % CG
        end
    end
    rf_all=rf;
    rf = reshape(rf(:,end),[length(k) Nc]); % qpwls_pcg returns all iterates
    
    toc
    % calculate excitation pattern
    m = zeros(dimb1);m(mask) = A*rf(:);
    
    err = norm(col(mask.*(abs(m)-abs(d))))/norm(col(d.*mask))
    
    [a,b]=blochsim_optcont_mex(ones(Nt,1),ones(Nt,1),rf*flipAngle/180*pi,sensBloch,garea,xx,omdt,1); %rf*dt*4258*2*pi
    
    Mxy=embed(2*conj(a).*b,Mask3dxyz128);
    Mz=embed(1-2*abs(b).^2,Mask3dxyz128);
    
    Mxy_RMSE = sqrt(mean((abs(Mxy(Mask3dxyz128_ring))-abs(d_smooth(Mask3dxyz128_ring))).^2))
    
    Mz_RMSE = sqrt(mean((abs(Mz(Mask3dxyz128_ring))-abs(d_ori(Mask3dxyz128_ring)-1)).^2))
    
    %err = norm(col(Mask3dxyz128_ring.*(abs(Mxy)-abs(d_smooth))))/norm(col(d_smooth.*Mask3dxyz128_ring))
    %err = norm(col(Mask3dxyz128.*(abs(Mxy)-abs(d_smooth))))/norm(col(d_smooth.*Mask3dxyz128))
    
    errmap=abs(abs(Mxy)-abs(d_smooth)).*Mask3dxyz128_ring;
    map_spatial=zeros(192,128);
    map_spatial(1:88,1:128)=squeeze(errmap(21:108,:,48));
    map_spatial(89:end,1:64)=squeeze(flip(errmap(64,21:124,21:84),3));
    map_spatial(97:184,65:end)=squeeze(flip(errmap(21:108,72,21:84),3));
    
    figure
    subplot(1,2,1)
    im(map_spatial,[0 0.1]);axis off;colorbar;title('Spatial Design');
    subplot(1,2,2)
    im(map,[0 0.1]);axis off;colorbar;title('K-space Design');
    
end