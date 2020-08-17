#ifndef NOTHREADS
#include <pthread.h> /* for threading */
#endif
#include <unistd.h> /* for sleep */
#include <sys/types.h>
#include <math.h>
#include "mex.h"

typedef struct {
    int ns,nc,nr,cmplxbrf;
    double *x,*y,*z;
    double *ar,*ai,*br,*bi;
    double *rfwr,*rfwi;
    double *sensr,*sensi;
    double *gareax,*gareay,*gareaz;
    double *omdt;
    double *brfr,*brfi;
    int *ntrf;
    int nstot;
    int nd;
} simparams;


void *blochsim_optcont(void *arg)
{
    
    simparams *p = (simparams *)arg;
    int i,j,k; /* counters */
    double ar,ai,br,bi;
    double sensr,sensi;
    double sensrfr,sensrfi;
    double art,ait,brt,bit;
    double zomr,zomi,zgr,zgi;
    double *brfr,*brfi;
    double *rfwr,*rfwi;
    double gtotal;
    double b1r,b1i,b1m;
    double c,s,sr,si;
    int cmplxbrf;
    int brfc;
    int nstot;
    int nr;
    int nd;
    nstot = p->nstot;
    nr = p->nr;
    nd = p->nd;
    cmplxbrf = p->cmplxbrf;
    rfwr = p->rfwr;
    rfwi = p->rfwi;
    brfr = p->brfr;
    brfi = p->brfi;
    
    for(i = 0;i < p->ns;i++){ /* loop over spatial locs */
        
        /* initialize */
        ar = 1;ai = 0;br = 0;bi = 0;
        
        zomr = cos(p->omdt[i]/2);
        zomi = sin(p->omdt[i]/2);
        
        brfc = 0;
        
        for(j = 0;j < p->nr;j++){ /* loop over rungs */
            
            /* get shimmed b1 for this rung */
            
            sensrfr = 0;sensrfi = 0;
            for(k = 0;k < p->nc;k++){
                sensr = p->sensr[i+k*nstot];
                sensi =  p->sensi[i+k*nstot];
                sensrfr += sensr*rfwr[j+k*nr] - sensi*rfwi[j+k*nr];
                sensrfi += sensi*rfwr[j+k*nr] + sensr*rfwi[j+k*nr];
            }
            
            for(k = 0;k < p->ntrf[j];k++){ /* loop over subpulse time */
                
                /* weight rf by subpulse */
                b1r = brfr[brfc+k]*sensrfr;
                b1i = brfr[brfc+k]*sensrfi;
                if(cmplxbrf == 1){
                    b1r -= brfi[brfc+k]*sensrfi;
                    b1i += brfi[brfc+k]*sensrfr;
                }
                
                /* apply RF */
                b1m = sqrt(b1r*b1r + b1i*b1i);
                c = cos(b1m/2);
                s = sin(b1m/2);
                if(b1m > 0){
                    sr = -s*b1i/b1m;
                    si = s*b1r/b1m;
                }else{
                    sr = 0;si = 0;
                }
                art = ar*c - br*sr - bi*si;
                ait = ai*c - bi*sr + br*si;
                brt = ar*sr - ai*si + br*c;
                bit = ar*si + ai*sr + bi*c;
                ar = art;ai = ait;
                br = brt;bi = bit;
                
                /* apply off resonance */
                art = ar*zomr - ai*zomi;
                ait = ai*zomr + ar*zomi;
                ar = art;ai = ait;
                brt = br*zomr + bi*zomi;
                bit = bi*zomr - br*zomi;
                br = brt;bi = bit;
                
            }
            brfc += p->ntrf[j];
            
            /* apply gradient blip */
            gtotal = p->x[i]*p->gareax[j] + p->y[i]*p->gareay[j];
            if(nd == 3) gtotal += p->z[i]*p->gareaz[j];
            zgr = cos(gtotal/2);
            zgi = sin(gtotal/2);
            art = ar*zgr - ai*zgi;
            ait = ai*zgr + ar*zgi;
            ar = art;ai = ait;
            brt = br*zgr + bi*zgi;
            bit = bi*zgr - br*zgi;
            br = brt;bi = bit;
            
        }
        
        /* copy into result */
        p->ar[i] = ar;
        p->ai[i] = ai;
        p->br[i] = br;
        p->bi[i] = bi;
        
    }
    
#ifndef NOTHREADS
    pthread_exit(NULL);
#endif
}

void mexFunction (int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    
    
    double *brfr,*brfi,*rfwr,*rfwi,*sensr,*sensi,*garea,*xx,*omdt;
    int nt,nd,nc,ns,nr,nttot; /* nthreads,ndims,ncoils,nspatiallocs */
    int *nst; /* number of spatial locs in each thread */
    double *ntrf; /* number of time points in each subpulse */
    int i,j; /* counters */
    double *ar,*ai,*br,*bi; /* pointers to real/imag output alpha and beta */
    simparams *p; /* structures passed to sim threads */
#ifndef NOTHREADS
    pthread_t *threads;
#endif
    int rc;
    int cmplxbrf;
    
    if (nrhs != 8)
        mexErrMsgTxt("Input arguments: brf,ntrf,rfw,sens,garea,xx,omdt,nthreads");
    if (nlhs != 2)
        mexErrMsgTxt("Output arguments: a,b");
    
    /* process input args */
    brfr = mxGetPr(prhs[0]);
    cmplxbrf = mxIsComplex(prhs[0]);
    if(cmplxbrf == 1)
        brfi = mxGetPi(prhs[0]);
    ntrf = mxGetPr(prhs[1]);
    rfwr = mxGetPr(prhs[2]);
    if(mxIsComplex(prhs[2]) == 0)
        mexErrMsgTxt("rfw must be complex.");
    rfwi = mxGetPi(prhs[2]);
    sensr = mxGetPr(prhs[3]);
    if(mxIsComplex(prhs[3]) == 0)
        mexErrMsgTxt("sens must be complex.");
    sensi = mxGetPi(prhs[3]);
    garea = mxGetPr(prhs[4]);
    xx = mxGetPr(prhs[5]);
    omdt = mxGetPr(prhs[6]);
    
    /* scalars */
    if(mxGetM(prhs[1]) > 1 & mxGetN(prhs[1]) > 1)
        mexErrMsgTxt("ntrf must be a vector");
    nr = mxGetM(prhs[1]) > mxGetN(prhs[1]) ? mxGetM(prhs[1]) : mxGetN(prhs[1]); /* number of rungs */
    nc = mxGetN(prhs[2]); /* number of coils */
    ns = mxGetM(prhs[3]); /* number of spatial locs */
    nd = mxGetN(prhs[4]); /* number of spatial dims */
    nt = mxGetScalar(prhs[7]); /* number of threads */
    nttot = mxGetM(prhs[0]); /* total number of time points */
    
    /* check # of dims */
    if (nd != mxGetN(prhs[5]))
        mexErrMsgTxt("Column dims of garea and xx must be equal.");
    /*  if(nd != 3)
      mexErrMsgTxt("Must be three spatial dimensions"); */
    /* check # of spatial locs */
    if ((ns != mxGetM(prhs[5])) || (ns != mxGetM(prhs[6])) ||
            (mxGetM(prhs[5]) != mxGetM(prhs[6])))
        mexErrMsgTxt("Row dims of sens, xx and omdt must be equal.");
    /* check # of coils */
    if (nc != mxGetN(prhs[3]))
        mexErrMsgTxt("Column dims of rfw and sens must be equal.");
    
    /* determine number of spatial locations per thread */
    nst = mxCalloc(nt,sizeof(int));
    for(i = 0;i < nt;i++) nst[i] = (int)(ns/nt);
    if(ns - ((int)(ns/nt))*nt > 0) /* need to add remainder to last thread */
        nst[nt-1] += ns - ((int)(ns/nt))*nt;
    
    /* allocate space for solutions */
    plhs[0] = mxCreateDoubleMatrix(ns,1,mxCOMPLEX);
    ar = mxGetPr(plhs[0]);
    ai = mxGetPi(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(ns,1,mxCOMPLEX);
    br = mxGetPr(plhs[1]);
    bi = mxGetPi(plhs[1]);
    
    /* allocate parameter space, copy p into it */
    p = mxCalloc(nt,sizeof(simparams));
    for(i = 0;i < nt;i++){
        
        /* scalars */
        p[i].ns = nst[i];
        p[i].nstot = ns;
        p[i].nc = nc;
        p[i].nr = nr;
        p[i].cmplxbrf = cmplxbrf;
        p[i].nd = nd;
        
        /* pointers to spatial locations */
        p[i].x = &xx[i*((int)(ns/nt))];
        p[i].y = &xx[i*((int)(ns/nt))+ns];
        if(nd == 3)
            p[i].z = &xx[i*((int)(ns/nt))+2*ns];
        
        /* pointers to sensitivities */
        p[i].sensr = &sensr[i*((int)(ns/nt))];
        p[i].sensi = &sensi[i*((int)(ns/nt))];
        
        /* pointer to off-resonance */
        p[i].omdt = &omdt[i*((int)(ns/nt))];
        
        /* pointers to output variables */
        p[i].ar = &ar[i*((int)(ns/nt))];
        p[i].ai = &ai[i*((int)(ns/nt))];
        p[i].br = &br[i*((int)(ns/nt))];
        p[i].bi = &bi[i*((int)(ns/nt))];
        
        /* copy brf */
        p[i].brfr = mxCalloc(nttot,sizeof(double));
        if(cmplxbrf == 1)
            p[i].brfi = mxCalloc(nttot,sizeof(double));
        for(j = 0;j < nttot;j++){
            p[i].brfr[j] = brfr[j];
            if(cmplxbrf == 1)
                p[i].brfi[j] = brfi[j];
        }
        
        /* copy ntrf */
        p[i].ntrf = mxCalloc(nr,sizeof(int));
        for(j = 0;j < nr;j++){
            p[i].ntrf[j] = ntrf[j];
        }
        
        /* copy rfw */
        p[i].rfwr = mxCalloc(nc*nr,sizeof(double));
        p[i].rfwi = mxCalloc(nc*nr,sizeof(double));
        for(j = 0;j < nc*nr;j++){
            p[i].rfwr[j] = rfwr[j];
            p[i].rfwi[j] = rfwi[j];
        }
        
        /* copy garea */
        p[i].gareax = mxCalloc(nr,sizeof(double));
        p[i].gareay = mxCalloc(nr,sizeof(double));
        if(nd == 3)
            p[i].gareaz = mxCalloc(nr,sizeof(double));
        for(j = 0;j < nr;j++){
            p[i].gareax[j] = garea[j];
            p[i].gareay[j] = garea[j+nr];
            if(nd == 3)
                p[i].gareaz[j] = garea[j+2*nr];
        }
        
    }
    
#ifndef NOTHREADS
    threads = mxCalloc(nt,sizeof(pthread_t));
    for(i=0; i < nt ; i++){
        rc = pthread_create(&threads[i], NULL, blochsim_optcont, (void *)&p[i]);
        /*    rc = 0;*/
        if (rc){
            mexErrMsgTxt("problem with return code from pthread_create()");
        }
    }
    
    /* wait for all threads to finish */
    for(i=0; i<nt;i++){
        pthread_join(threads[i],NULL);
    }
#else
    /* process them serially */
    for(i=0; i < nt; i++) blochsim_optcont((void *)&p[i]);
#endif
    
    /* release memory */
    for(i=0; i<nt;i++){
        /* release brf */
        mxFree(p[i].brfr);
        if(cmplxbrf == 1)
            mxFree(p[i].brfi);
        /* release ntrf */
        mxFree(p[i].ntrf);
        /* release rfw */
        mxFree(p[i].rfwr);
        mxFree(p[i].rfwi);
        /* release garea */
        mxFree(p[i].gareax);
        mxFree(p[i].gareay);
        if(nd == 3)
            mxFree(p[i].gareaz);
    }
    mxFree(nst);
    mxFree(p);
#ifndef NOTHREADS
    mxFree(threads);
#endif
    
    return;
}

