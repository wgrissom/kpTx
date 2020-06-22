// Important
// Compared to the old method we developed
// The NhN matrix here is the StS matrix in our old method
// The NhC matrix here is the conj(Srhs) matrix in our old method
// coeff is w in our old method, which are the non-zero elements of columns of W

/* LS_fft_mex.c
 *
 * compile with "mex -largeArrayDims -lmwlapack LS_fft_mex_clean.c"
 *
 * As the FFT trick grid may be large, here we enforce using
 *   size_t:    for (large-sized) grid indexing
 *   ptrdiff_t: to address negative offsets from grid center
 * instead of mwSize and mwSignedIndex
 *
 * some acronyms used:
 * _ctor: constructor
 * _dtor: destructor
 * _B:    a Block of a block matrix
 * _P:    Pointer
 * _U:    Upper matrix
 * _L:    Lower matrix
 * _t:    Temporary (when appended to a variable)
 * _i:    Iterator
 * _R:    Real
 * _I:    Imag
 */
#if !defined(_WIN32)
#define cposv cposv_
/* #define chesv chesv_ */
#endif

#include <mex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lapack.h>
#include <stddef.h> /* common macros: size_t, ptrdiff_t, etc. */

/* def of GRID
 * GIM GCO GNO are defined w/ ptrdiff_t as they are used in signed index
 * calculation in LS_init()
 */
struct GRID
{
  float      Tik;  /* (1,), Tik regularizor coeff */
  size_t     nC;   /* (1,), number of coils */
  size_t     nSolve;   /* (1,), number of solved points (delta function) */
  size_t     nDim; /* (1,), number of dimensions (1|2|3) */
  size_t     nVtx; /* (1,), number of lin-interp ngbs, (2|4|8) */
  mwSize    *nd;   /* (nDim,), grid size along each dim */
  mxArray   *G;    /* (nC*(nC+1)/2,), Grid cells */
  mxArray   *G_NhC;    /* (nC), Grid cells */
  float     *GIU;  /* (nDim,), _Inv _Unit, normalizing shifts for grids */
  ptrdiff_t *GIM;  /* (nDim,), _Index _Multiplier */
  ptrdiff_t  GCO;  /* (1,),    _Center _Offset */
  ptrdiff_t *GNO;  /* (nVtx,), _Neighbors _Offsets */
};

// GIU is idgrid, G is F_c
struct GRID *Grid_ctor(
  const size_t   nDim,
  const size_t   nC,
  const size_t   nSolve,
  const float    Tik,
  const float   *GIU,
  const mxArray *G,
  const mxArray *G_NhC)
{
  struct GRID *grid = mxCalloc(1, sizeof(struct GRID));

  size_t nVtx_t = 2, i_t;
  for (i_t = 1; i_t < nDim; i_t++) {nVtx_t *= 2;}
  const size_t nVtx = nVtx_t;

  grid->Tik  = Tik;
  grid->nC   = nC;
  grid->nSolve   = nSolve;
  grid->nDim = nDim;
  grid->nVtx = nVtx;

  grid->nd = mxCalloc(nDim, sizeof(size_t));
  memcpy(grid->nd, mxGetDimensions(mxGetCell(G,0)), nDim*sizeof(mwSize));
  // mexGetCell(G,0) Pointer to the first (index 0 is first) element in cell array
  // G is a cell of mxArrays, mxGetCell(G,0) points to one of the element-mexArray
  
  // mxGetDimensions returns
  // Pointer to the first element in the dimensions array. 
  // Each integer in the dimensions array represents the number of elements in a particular dimension.  
  
  // nd is the vector of dimension of elements in F_c.
  // which is the oversampled FFT of the coil-combination-image
  
  grid->G  = (G);
  grid->G_NhC  = (G_NhC);
  
  // In order to let G and G_NhC lives between mex function calls (make persistent)
  // They have to be duplicated. 
  // I stopped doing this in order to safe memory.
  //grid->G  = mxDuplicateArray(G);
  //grid->G_NhC  = mxDuplicateArray(G_NhC);

  grid->GIU = mxCalloc(nDim, sizeof(float));
  memcpy(grid->GIU, GIU, nDim*sizeof(float));
  grid->GIM = mxCalloc(nDim, sizeof(ptrdiff_t));
  grid->GCO = 0;
  grid->GNO = mxCalloc(nVtx, sizeof(ptrdiff_t));

  const mwSize *nd = grid->nd;
  /* setup GCO GNO GIM */
  ptrdiff_t *GCO, *GNO, *GIM;
  GCO = &(grid->GCO);
  GNO =   grid->GNO + nVtx-1;
  GIM =   grid->GIM + nDim;
  //GCO,GNO,GIM are all pointers pointing places close to grid->GCO,grid->GNO,grid->GIM
  //So we can use GCO, GNO, GIM to give values to there correspondents in the object "grid", 
  //without manupilating pointers in the object "grid"
 
  
  
      // Cheat sheet: 
        // A++ is runing ++ after this line ends
        // ++A is runing ++ before this line begins
        // *A++ is finding the value A points to,then increase A after finishing this line
        // *++A is first doing ++ before the whole line, then finds the value (A after increament) points to, 
 
  switch(nDim)
  {           
    //This switch is just to choose a start point, it will run all the way down
    case 3:
      *GCO  += ( (*(nd+2))/2 ) * (*(nd+1)) * (*nd);
      *--GIM = *(nd+1) * (*nd);
      *GNO-- = *GIM + *nd + 1;
      *GNO-- = *GIM + *nd;
      *GNO-- = *GIM + 1;
      *GNO-- = *GIM; /* no break */
    case 2:
      *GCO  += ( (*(nd+1))/2 ) * (*nd);
      *--GIM = *nd;
      *GNO-- = *nd + 1;
      *GNO-- = *nd; /* no break */
    case 1:
      *GCO  += ( (*nd)/2+!!(*nd) ) - 1; /* -1 convert to 0-based indexing */
      *--GIM = 1;
      *GNO-- = 1;
      *GNO   = 0; /* break only in case 1 */
      break;
    default:
      mexErrMsgTxt("Only support nDim == 1 or 2 or 3");
  } /* [0, 1(, nx, nx+1(, ny*nx, ny*nx+1, ny*nx+nx, ny*nx+nx+1))] */
  
  // At the end of this switch, GCO becomes the index of the center element in F_c
  // Again, F_c is the oversampled FT maps of the coil combination images
  
  // GIM is the index multiplier for each dimension
  
  //GNO are the shifts (or relative indexing) for interpolation sample points
  //Note they are upper-left,lower-left, upper-right, lower-right, (square shape)
  //Instead of left, right, up down (cros shape)
  

  return grid;
}

void Grid_dtor(struct GRID *grid)
{
  mxFree(grid->nd);

  //mxDestroyArray(grid->G);
  //mxDestroyArray(grid->G_NhC);

  mxFree(grid->GIU);
  mxFree(grid->GIM);
  mxFree(grid->GNO);

  mxFree(grid);
}

/* def of LS
 * NhN_GI NhC_GI are defined w/ size_t, as grid may be large.
 * Their calculation involves signed addition
 */
struct LS
{
  //size_t  nNgb;

  float  *NhN;       /* (2*nC*nNgb, nC*nNgb), 2 for complex */
  float  *NhC;       /* (2*nC*nNgb, nSolve), 2 for complex */
  // in kspace PTX, nSolve = SegWidth^ndim
  
  /* _Block _Ptr, pointers indexing the blocks of NhN & NhC */
  float **NhN_BPL; /* L->(nC*(nC+1)/2,): L-triangle w/ Diag */
  float **NhN_BPU; /* U->(nC*(nC-1)/2,): U-triangle */
  float **NhC_BPL;

  /* _Interpolation _Coefficients (_Reversed ptr) */
  float  *IC;        /* (nVtx,) container for interpolation coeffs */
  float  *GD;        /* (nDim,) container for floored _Grid _Distance */
};

struct LS *LS_ctor(const struct GRID *grid)
{
  struct LS *ls = mxCalloc(1, sizeof(struct LS));

  const size_t nC = grid->nC;
  const size_t nSolve = grid->nSolve;

  ls->NhN = NULL;
  ls->NhC = NULL;
 
  ls->NhN_BPL = mxCalloc((nC*(nC+1))/2, sizeof(float*));
  ls->NhN_BPU = mxCalloc((nC*(nC-1))/2, sizeof(float*));
  ls->NhC_BPL = mxCalloc(nC, sizeof(float*));

  ls->IC = mxCalloc(grid->nVtx, sizeof(float));
  ls->GD = mxCalloc(grid->nDim, sizeof(float));

  return ls;
}

/* prepare block indices */
void LS_BP_prep(
  const size_t  nNgb,
  const size_t  nC,
  struct LS    *ls)
{
  /* setup ls->NhN_BP & ls->NhC_BP, 2 for Fortran */
  const size_t nNgb2     = 2*nNgb;
  const size_t nNgbSq2   = nNgb*nNgb2;
  const size_t nCnNgbSq2 = nC*nNgbSq2;  //nC*nNgb*nNgb*2

  size_t  CoilRef_i, CoilQry_i;
  float **NhN_BPL_t = ls->NhN_BPL;
  float **NhN_BPU_t = ls->NhN_BPU;
  float **NhC_BPL_t = ls->NhC_BPL;
  float  *NhN_Lt, *NhC_Lt, *NhN_Ut; /* _Lower & _Upper triangle _tmp */ 
  
  for (CoilRef_i = 0; CoilRef_i < nC; CoilRef_i++)
  {
    /* _Upper triangle _temp current value is not used, just to update offset */
    *NhN_BPL_t++ = (NhN_Ut = NhN_Lt = ls->NhN + CoilRef_i*(nCnNgbSq2+nNgb2));        
    //Do the diagnal blocks first
    
    *NhC_BPL_t++ = ls->NhC + CoilRef_i*(nNgb2); //NhC_BPL for kptx
    
    //For Lower triangle, CoilRef is column, CoilQry is row
    //For Upper triangle,  CoilRef is row, CoilQry is column
    for (CoilQry_i = CoilRef_i+1; CoilQry_i < nC; CoilQry_i++)
    {
      *NhN_BPL_t++ = (NhN_Lt += nNgb2);     

      *NhN_BPU_t++ = (NhN_Ut += nCnNgbSq2);
      
    }
  }
}

void LS_init(
  const size_t       nNgb,
  const size_t       nSolve,
  const struct GRID *grid,
        struct LS   *ls)
{
  const size_t nC = grid->nC;
  ls->NhN  = mxCalloc(2*(nC*nNgb)*(nC*nNgb), sizeof(float)); /* 2 for cmplx */
  ls->NhC  = mxCalloc(2*(nC*nNgb)*nSolve,        sizeof(float));

  LS_BP_prep(nNgb, nC, ls);
}

/* This only frees some struct members */
/* Double free for this struct is allowed, as it has member being persistent */
void LS_free(struct LS *ls)
{
  if(NULL != ls->NhN)
  {
    mxFree(ls->NhN);
    ls->NhN = NULL;
  }
  if(NULL != ls->NhC)
  {
    mxFree(ls->NhC);
    ls->NhC = NULL;
  }
}

/* Double destruct is not allowed tho */
void LS_dtor(struct LS *ls)
{
  /* LS_free(ls); */ /* non- var are taken cared by matlab */

  mxFree(ls->NhN_BPL);
  mxFree(ls->NhN_BPU);
  mxFree(ls->NhC_BPL);

  mxFree(ls->IC);
  mxFree(ls->GD);

  mxFree(ls);
}

static struct GRID *grid;
static struct LS   *ls;
static int          constructed = 0;


void freePersistent(void)
{
  Grid_dtor(grid);
  LS_dtor(ls);
}

/*
 * core part
 */
//diff  = (GIU[i]) * (shRef[i] - shQry[i]);
//GD[i] = diff - floorf(diff);  //GD only save the fractional part (after digit point)
//*GI  += (GIM[i]) * (ptrdiff_t)floorf(diff);

// do interp using coefficients
//{ for(i=0;i<nVtx;i++) { res += *(IC++) * Gdata[GI0 + *(GNO++)]; } }
// If we want to interp the midpoint C between point A and B
// It will be (a+b)/2, here 1/2 is the coefficient of A and B
// When C move toward A or B, the coefficient will incerase or decrease C with respect to the distance
// That's why GD only save the fractional part

//But where is the denominator? for 1D is (x1-x2),for 2D is (x1-x2)(y1-y2)
void IC_calc(
  const size_t  nVtx, /* (1,), # of vertices used for interpolation 2/4/8 */
        float  *GD,   /* (nDim,), shortest Grid Distance of each _dim */
        float  *IC)   /* (nVtx,), storing the coefficients*/
{
  switch(nVtx)
  {
    case 2:
      *IC++ =                         (1-*GD);
      *IC   =                         (  *GD);
      break;
    case 4:
      //Using bilinear interpolation
      *IC++ =             (1-*(GD+1))*(1-*GD);
      *IC++ =             (1-*(GD+1))*(  *GD);
      *IC++ =             (  *(GD+1))*(1-*GD);
      *IC   =             (  *(GD+1))*(  *GD);
      break;
    case 8:
      *IC++ = (1-*(GD+2))*(1-*(GD+1))*(1-*GD);
      *IC++ = (1-*(GD+2))*(1-*(GD+1))*(  *GD);
      *IC++ = (1-*(GD+2))*(  *(GD+1))*(1-*GD);
      *IC++ = (1-*(GD+2))*(  *(GD+1))*(  *GD);
      *IC++ = (  *(GD+2))*(1-*(GD+1))*(1-*GD);
      *IC++ = (  *(GD+2))*(1-*(GD+1))*(  *GD);
      *IC++ = (  *(GD+2))*(  *(GD+1))*(1-*GD);
      *IC   = (  *(GD+2))*(  *(GD+1))*(  *GD);
      break;
    default:
      mexErrMsgTxt("Only support nVtx == 2 or 4 or 8");
  }
}

//Ls_NhN_form fill each Ai^H*Aj(ith and jth coil combination) blocks one by one
//(every round, one block in upper trangle and one block in lower triangle will be fill together)

//For blocks in lower triangle, it does row first, then change to next row 
//For blocks in upper triangle, it does column first, then change to next column



// *GNO;  _Neighbors _Offsets
float doInterp(
  const float     *Gdata,
  const size_t     GI0,
  const ptrdiff_t *GNO,
  const float     *IC,
  const size_t     nVtx,
  const int        isRev)
{
  float res = 0;
  size_t i;
  if (isRev) { for(i=0;i<nVtx;i++) { res += *(IC++) * Gdata[GI0 - *(GNO++)]; } }
  else       { for(i=0;i<nVtx;i++) { res += *(IC++) * Gdata[GI0 + *(GNO++)]; } }
  return res;
}


//LS_NhN_form calls it this way
//update_GDGI(nDim, GIU, GIM, shQry_t, shRef, GD, &GI);
//LS_NhC_form calls it this way
//update_GDGI(nDim, GIU, GIM, shQry_t, NULL, GD, &GI);      

// shQry and shQry are all in the unit before dense sampling
// GI is the index difference (shRef - shQry) in the dense sampled grid
// GD is the fractional part of the difference in the dense sampled grid
void update_GDGI(
  const size_t     nDim,
  const float     *GIU,
  const ptrdiff_t *GIM,
  const float     *shQry,
  const float     *shRef,
        float     *GD,
        ptrdiff_t *GI)
{
  float  diff;
  size_t i = 0;
  *GI = 0;
  if (NULL == shRef)  //This part is for NhC
  {
    for(i = 0; i < nDim; i++)
    {
      diff  = (GIU[i]) * (         - shQry[i]);
      GD[i] = diff - floorf(diff); 
      *GI  += (GIM[i]) * (ptrdiff_t)floorf(diff);
    }
  }
  else   //This part is for NhN
  {
    for(i = 0; i < nDim; i++)
    {
      diff  = (GIU[i]) * (shRef[i] - shQry[i]);
      GD[i] = diff - floorf(diff);  //GD only saves fractions (after digit point)
      *GI  += (GIM[i]) * (ptrdiff_t)floorf(diff);
      //shRef[i]-shQry[i] is in the unit of 1/FOV, which is before dense padding
      //diff = (GIU[i]) * (shRef[i] - shQry[i]) changed the unit into indecies after dense padding
      //GI is the index differce between these two poins.
      //sh is sampled-unsample, diff is sampledREF-sampleQRY
    }
  }
}

void LS_NhN_form(
  const size_t       nNgb,
  const float       *sh,
  const struct GRID *grid,
        struct LS   *ls)
{
  const float     *shRef   = sh;
  const float     *shQry_t = NULL; /* needs reset for each Ref */
  /* Para from grid */
  const float     *GIU = grid->GIU;
  const ptrdiff_t *GIM = grid->GIM;
  const ptrdiff_t *GNO = grid->GNO;
  const mxArray   *G   = grid->G;
  const float     *G_R, *G_I;
  const float      Tik = grid->Tik;
  const size_t nC = grid->nC, nDim = grid->nDim, nVtx = grid->nVtx;

  /* interp containers: coefficients, grid_dist, grid_ind */
  float *IC = ls->IC, *GD = ls->GD;
  ptrdiff_t GI = 0;  /* _Grid _Ind to the 1st neighbor for interp */
  size_t    GI0 = 0; /* _Grid _Ind, starting at 0 */

  /* BPL & BPU iteraters */
  size_t   inc = 2*nC*nNgb - 1; /* 2 for cplx, -1 shift from imag to real */
  size_t   NhN_BPL_i, NhN_BPU_i;
  float  **NhN_BPL_t = ls->NhN_BPL, **NhN_BPU_t = ls->NhN_BPU;

  /* for-loop vars */
  size_t   NgbRef_i, NgbQry_i, CoilRef_i, CoilQry_i;
  float    R, I;

    NhN_BPL_i = NhN_BPU_i = 0;

    for(CoilRef_i = 0; CoilRef_i < nC; CoilRef_i++)
    {
        for(CoilQry_i = CoilRef_i; CoilQry_i < nC; CoilQry_i++)
        {
            G_R = (const float*)    mxGetData(mxGetCell(G, NhN_BPL_i));
            G_I = (const float*)mxGetImagData(mxGetCell(G, NhN_BPL_i)) ;

          shRef   = sh;

          // NgbRef is column for lower triangle blocks, and row for upper triangle blocks
          // NgbQry is row for lower tirangle blocks, and column for upper tirangle blocks
          for(NgbRef_i = 0; NgbRef_i < nNgb; NgbRef_i++)
          {
            shQry_t = sh;     /* reset to the 1st shift  */
            for(NgbQry_i = 0; NgbQry_i < nNgb; NgbQry_i++)
            {
              //GIU is the inverse unit, it is just two single (in 2D case)
              //GIM is the index multiplier for changing plane or col
              //GI come here as 0
              //GD is only distributed memory in LS constructor, but not given value  
              update_GDGI(nDim, GIU, GIM, shQry_t, shRef, GD, &GI);
              //diff  = (GIU[i]) * (shRef[i] - shQry[i]);
              //GD[i] = diff - floorf(diff);  //GD only saves fractional (after digit point)
              //*GI  += (GIM[i]) * (ptrdiff_t)floorf(diff);

              GI0 = grid->GCO + GI; /* center offset */
              // GCO is the center intext of elements in F_c
              // which is the oversampled FFT of the coil combination image
              // GI is the index difference shQry-shQry?in dense sampled matrix

              /* update coefficients & do interpolation */
              IC_calc(nVtx, GD, IC);

              *(NhN_BPL_t[NhN_BPL_i]++) = (R=doInterp(G_R, GI0, GNO, IC, nVtx, 0));
              *(NhN_BPL_t[NhN_BPL_i]++) = (I=doInterp(G_I, GI0, GNO, IC, nVtx, 0));
              // dointerp is using the points around GIO to do the interpolation

              if (CoilRef_i != CoilQry_i)
              {
                  *(NhN_BPU_t[NhN_BPU_i]++) = R;
                  *(NhN_BPU_t[NhN_BPU_i]) = -I; /* pointing in-row next element */
                  NhN_BPU_t[NhN_BPU_i] += inc;
                  //NhN_L is doing a column, so the increament is just one
                  //NhN_U is correspondingly doing a row, so it needs "inc"
                  //inc = 2*nC*nNgb - 1; /* 2 for cplx, -1 shift from imag to real */
              }
              //NhN_BPL_t stores a set of pointers, one pointer for a lower triangle block
              //each pointer points to one location in the large A^H*A matrix 
              //this location is indexed according to A^H*A instead of Ai^h*Aj^h, so we need "inc" and NhN_BP_NextRef

              //But one pointer's pointing location will only be within its corresponding lower triganlge blcok

              //And the the indexing of the pointers within NhN_BPL_t are just one increament, no offset needed.

              shQry_t += nDim; /* next query shift */
            }
            
            //here we change column/row for one lower/upper triangle block
            NhN_BPL_t[NhN_BPL_i] += 2*((nC-1)*nNgb);
            if (CoilRef_i != CoilQry_i)
            {NhN_BPU_t[NhN_BPU_i] -= 2*(nC*nNgb*nNgb - 1);}

            shRef += nDim;     /* next reference shift */
          }
          NhN_BPL_i++;
          if (CoilRef_i != CoilQry_i)
          {NhN_BPU_i++;}
        }        
    }
  /* apply the Tik reg */
  size_t Tik_i = 0, Tik_inc = 2*(nC*nNgb+1), nTik = nC*nNgb;
  float  *NhN_t = ls->NhN, nNgbTik = Tik;//nNgbTik = nNgb*Tik;
  if (Tik)
  {
    for(Tik_i = 0; Tik_i < nTik; Tik_i++)
    {
      *NhN_t += nNgbTik;
      NhN_t  += Tik_inc;
    }
  }
}

// In the context of grappa, NhC has coil-number of columns, 
// because each column has its own ACS data and also its missing points to fill.
// But in the context of kPtx, NhC only has one column, 
// because there is only one "total" excitation k-sapce to be fill.
void LS_NhC_form(
  const size_t       nNgb,
  const size_t       nSolve,
  const float       *sh,
  const float       *shSolve,
  const struct GRID *grid,
        struct LS   *ls)
{
  const float     *shQry_t = sh;
  const float     *shSolve_t = shSolve; //shifts for solving points
  /* Paras from grid */
  const float     *GIU = grid->GIU;
  const ptrdiff_t *GIM = grid->GIM;
  const ptrdiff_t *GNO = grid->GNO;
  const mxArray   *G_NhC = grid->G_NhC;
  const float     *G_R, *G_I;
  const size_t nC = grid->nC, nDim = grid->nDim, nVtx = grid->nVtx;

  /* interp containers: coefficients, grid_dist, grid_ind */
  float *IC = ls->IC, *GD = ls->GD;
  ptrdiff_t GI = 0;  /* _Grid _Ind to the 1st neighbor for interpolation */
  size_t    GI0_L = 0, GI0_U = 0; /* _Grid _Ind, starting at 0 */

  /* BPL & BPU iteraters */
  size_t   NhC_BPL_i;  
  float  **NhC_BPL_t = ls->NhC_BPL; 


  /* for-loop vars */
  size_t  CtrQry_i = 0, SolveQry_i = 0, CoilRef_i = 0, CoilQry_i = 0;
  
  NhC_BPL_i = 0; 
  for(CoilRef_i = 0; CoilRef_i < nC; CoilRef_i++)
  {
      G_R = (const float*)    mxGetData(mxGetCell(G_NhC, NhC_BPL_i));
      G_I = (const float*)mxGetImagData(mxGetCell(G_NhC, NhC_BPL_i));

      shSolve_t = shSolve;
      
      // SolveQry_i should be the column number of NhC
      for(SolveQry_i = 0; SolveQry_i < nSolve; SolveQry_i++)    
      {
          shQry_t = sh;     /* reset to the 1st shift  */
          for(CtrQry_i = 0; CtrQry_i < nNgb; CtrQry_i++)
          {
            update_GDGI(nDim, GIU, GIM, shSolve_t, shQry_t, GD, &GI);
            // Note here shQry_t switched place with Null(shSolve_t) compared to formNhC in grapaa
            // because what we want to interpolate is k_Solve-k_Traj
            // and the input sh is -(k_traj-k_targ), 
            //so when calculate diff in update_GDGI, shQry_t should be in the place of shRef

            // Also, here shQry should be the shift corresponding to the tall matrix S
            // instead of the wide matrix S^H, so 
            // (the big matrix NhN comes from S^H*S)

            // form NhC in grappa call it this way
            // update_GDGI(nDim, GIU, GIM, shQry_t, Null, GD, &GI);
            // form NhN call it this way, note Null vs shRef   
            //update_GDGI(nDim, GIU, GIM, shQry_t, shRef, GD, &GI);
            // in form NhN, update_GDGI has this
              //diff  = (GIU[i]) * (shRef[i] - shQry[i]);
              //GD[i] = diff - floorf(diff);  //GD only saves fractional (after digit point)
              //*GI  += (GIM[i]) * (ptrdiff_t)floorf(diff);

            GI0_L = grid->GCO + GI; /* nd-array center offset */


            /* update coefficients & do interpolation */
            IC_calc(nVtx, GD, IC);
            
              *(NhC_BPL_t[NhC_BPL_i]++) = doInterp(G_R, GI0_L, GNO, IC, nVtx, 0);
              *(NhC_BPL_t[NhC_BPL_i]++) = -doInterp(G_I, GI0_L, GNO, IC, nVtx, 0);
              // in kptx, Srhs needs to be conjugated

              

    // in kptx, NhC only loop through one set of coils, no coil combintation
    
            
            shQry_t += nDim; /* next query shift */
          }

        //here we change column/row for one coils block
            
        NhC_BPL_t[NhC_BPL_i] += 2*((nC-1)*nNgb); 

        shSolve_t += nDim; /* next shift for kSolve*/  
      }
      NhC_BPL_i++;
  }
     
}

/*
 * Convert Fortran complex storage to MATLAB real and imaginary parts.
 * X = fort2mat(Z,ldz,m,n) copies Z to X, producing a complex mxArray
 * with mxGetM(X) = m and mxGetN(X) = n.
 */
mxArray* fort2mat(
  const size_t    m,
  const size_t    n,
  const ptrdiff_t ldz,
        float    *Z)
{
  size_t i,j;
  ptrdiff_t incz;
  float *xr,*xi,*zp;
  mxArray *X;

  X = mxCreateNumericMatrix(m,n, mxSINGLE_CLASS, mxCOMPLEX);
  xr = (float *)mxGetData(X);
  xi = (float *)mxGetImagData(X);
  zp = Z;
  incz = 2*(ldz-m);

  for (j = 0; j < n; j++)
  {
    for (i = 0; i < m; i++)
    {
      *xr++ = *zp++;
      *xi++ = *zp++;
    }
    zp += incz;
  }

  return(X);
}

/*
 * Function [coeff] = LS_fft_mex(sh, nC, Tik, F_c, dgrid, shSolve, F_c_NhC)
 * INPUTS:
 *  - sh (nDim, nNgb) single, shifts, NOTICE the dimensions
 *  - ->Tik (1,), Tiknov regularizer
 *  - ->nC  (1,), number of coils
 *  - F_c (->nC*(->nC+1)/2,) cell, spectrum of conj correlated ACS
 *  - idgrid (nDim,) single, inv grid size for normalizing shifts to calc coefs
 *
 */
void mexFunction(
  int nlhs,       mxArray *plhs[],
  int nrhs, const mxArray *prhs[])
{
  
  if (!mxIsSingle(mxGetCell(prhs[0],0))) {mexErrMsgTxt("Require single shifts input");}
  const size_t nDim = mxGetM(prhs[5]);
  size_t nNgb; 
  const size_t nSolve = mxGetN(prhs[5]);
  const size_t nSeg = mxGetN(prhs[0]);

  /* handling INPUTS */
  //if( !mxIsSingle(prhs[0]) ) { mexErrMsgTxt( "sh: single precision" ); }
  float *sh;
  const float *shSolve = (float*)mxGetData(prhs[5]);

  size_t nC;
  float *GIU;
  float  Tik = (float) mxGetScalar(prhs[2]);
  //if (!constructed) /* dealing with the remaining prhs */
  {
    nC  = (size_t)mxGetScalar(prhs[1]);

    if( !mxIsSingle(prhs[4]) ) { mexErrMsgTxt( "dgrid: single precision" ); }
    GIU = (float *)mxGetData(prhs[4]);
    // GIU is idgrid, it just two single (in 2D case)
    
    /* Initialize the two essential structs */
    grid = Grid_ctor(nDim, nC, nSolve, Tik, GIU, prhs[3],prhs[6]); /* persistent static var */
    //GIU is idgrid (2 singles). prhs[3] is F_c, the oversampled FFT of the coil-combination-image
    ls   = LS_ctor(grid);

    

    constructed = 1;
  }
  //else { grid->Tik = Tik; }
  
  
  size_t nSeg_t, N,NRHS = nSolve; 
  float *NhN,*NhC;
  mxArray *coef_c = mxCreateCellMatrix((mwSize)nSeg,1);
  mxArray *coef_t;
 
  
  for (nSeg_t = 0; nSeg_t < nSeg; nSeg_t++)
  {
      if(mxGetN(mxGetCell(prhs[0],nSeg_t)) != 0)
      {
          
          sh = (float*)mxGetData(mxGetCell(prhs[0],nSeg_t));

          nNgb=mxGetN(mxGetCell(prhs[0],nSeg_t));
          LS_init(nNgb, nSolve, grid, ls); /* update ls by the input shifts */
          
          LS_NhN_form(nNgb, sh, grid, ls);
          LS_NhC_form(nNgb, nSolve, sh, shSolve, grid, ls);

          NhN = ls->NhN;
          NhC = ls->NhC;
          
          N = grid->nC*nNgb; 

          /* Call LAPACK function */
          char uplo = 'U';
          mwSignedIndex info;

          cposv(&uplo, &N, &NRHS, NhN, &N, NhC, &N, &info);

              /*
            http://www.netlib.org/lapack/explore-html/db/d91/group__complex_p_osolve_gad6fa5e367df37b944f5224b5dcc6ab50.html#gad6fa5e367df37b944f5224b5dcc6ab50
            subroutine cposv	(	character 	UPLO,
            integer 	N,
            integer 	NRHS,
            complex, dimension( lda, * ) 	A,
            integer 	LDA, (stnads for leading dimention)
            complex, dimension( ldb, * ) 	B,
            integer 	LDB,
            integer 	INFO 
            )	
              CPOSV computes the solution to a complex system of linear equations
                A * X = B,
             where A is an N-by-N Hermitian positive definite matrix and X and B
             are N-by-NRHS matrices.
                      B is COMPLEX array, dimension (LDB,NRHS)
                      On entry, the N-by-NRHS right hand side matrix B.
                      On exit, if INFO = 0, the N-by-NRHS solution matrix X.
             */

          if (info != 0) {mexPrintf("LS_fft_mex: ill-conditioned, try a larger Tik\n");}

          /* LAPACK always do in-place operation, so NhC is now coeffs */
          coef_t = fort2mat(N, NRHS, N, NhC);
          // N=nC*nNgb this is only the length of NhN and NhC
          // NRHS=nSolve=SegWidth^nDim
          
         // Since in the LS_free, NhC which is the same as the coef_t, will be disconnected from the corresponding memory,
         // so the coef_t that has been set to cell, won't be re-written.
         mxSetCell(coef_c,nSeg_t,(coef_t));
          

          LS_free(ls);
         
      }
  }
   /* clean up */
  
  freePersistent();
   
  plhs[0]=coef_c;
  
}

