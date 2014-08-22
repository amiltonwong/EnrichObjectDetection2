#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "cudaDecorrelateFeature.h"
#include "cudaDecorrelateFeature.cuh"

static bool debug = true;

    /*
        STAGE
                1. Return host Scrambled Sigma
                2. Return host 
    */

/*
    prhs[0] = gpu_Gamma     single precision
    prhs[1] = HOGTemplate   single precision
    prhs[2] = nonEmptyRows  int32
    prhs[3] = nonEmptyCols  int32
    prhs[4] = lambda        double
    prhs[5] = (optional) FEATURE_THRESHOLD
    prhs[6] = (optional) CG_TOLERANCE
    prhs[7] = (optional) CG_MAX_ITER
    prhs[8] = (optional) thread size
*/
////////////////////////////////////////////////////////////////////////////////
// Mex Entry
////////////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "cudaDecorrelateFeature:InvalidInput";

    float FEATURE_THRESHOLD = 1.5f;
    float CG_TOLERANCE      = 0.001f;
    int CG_MAX_ITER         = 60;
    int TEMPLATE_WIDTH, TEMPLATE_HEIGHT, FEATURE_DIM;

    /* Choose a reasonably sized number of threads for the block. */
    int THREAD_PER_BLOCK_H  = 16;
    int THREAD_PER_BLOCK_W  = 8;
    int THREAD_PER_BLOCK_D  = 8;
    int THREAD_PER_BLOCK_2D_H = 32;
    int THREAD_PER_BLOCK_2D_W = 32;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    int SIGMA_OUT = 0;

    int GAMMA_IN            = 0;
    int TEMPLATE_IN         = 1;
    int NON_EMPTY_ROW_IN    = 2;
    int NON_EMPTY_COL_IN    = 3;
    int LAMBDA_IN           = 4;
    int FEATURE_THRESHOLD_IN= 5;
    int CG_TOLERANCE_IN     = 6;
    int CG_MAX_ITER_IN      = 7;
    int THREAD_SIZE_IN      = 8;

    if ( (nrhs < 5) || (nrhs > 9) )
        mexErrMsgIdAndTxt(errId, "Wrong number of inputs");
    

    /* Gamma */
    if ( !mxIsGPUArray(prhs[GAMMA_IN]))
        mexErrMsgIdAndTxt(errId, "The Gamma must be real single precision array in GPU");
    const mxGPUArray *mxGamma = mxGPUCreateFromMxArray(prhs[GAMMA_IN]);
    if ( mxGPUGetClassID(mxGamma) != mxSINGLE_CLASS )
        mexErrMsgIdAndTxt(errId, "The Gamma must be real single precision array in GPU");
    const mwSize *mxGamma_Dim = mxGPUGetDimensions(mxGamma);
    const int GammaDim        = mxGamma_Dim[0];
    const float *d_Gamma      = (float *)mxGPUGetDataReadOnly(mxGamma);


    /* Template */
    const mxArray *mxHogTemplate = prhs[TEMPLATE_IN];
    if ( mxGetNumberOfDimensions(mxHogTemplate) != 3 || mxGetClassID(mxHogTemplate) != mxSINGLE_CLASS)
        mexErrMsgTxt("Invalid input: hog template");

    const mwSize *TEMPLATE_DIM = mxGetDimensions(mxHogTemplate);
    TEMPLATE_HEIGHT = TEMPLATE_DIM[0];
    TEMPLATE_WIDTH  = TEMPLATE_DIM[1];
    FEATURE_DIM     = TEMPLATE_DIM[2];


    /* Non Empty Col and Row Index */
    const mxArray *mxNonEmptyRows       = prhs[NON_EMPTY_ROW_IN];
    const mxArray *mxNonEmptyCols       = prhs[NON_EMPTY_COL_IN];
    const mwSize  *mxNonEmptyRowsDim    = mxGetDimensions(mxNonEmptyRows);
    const mwSize  *mxNonEmptyColsDim    = mxGetDimensions(mxNonEmptyCols);
    if( mxNonEmptyRowsDim[0] != mxNonEmptyColsDim[0] ||
        mxNonEmptyRowsDim[1] != mxNonEmptyColsDim[1] ||
        mxGetClassID(mxNonEmptyRows) != mxINT32_CLASS ||
        mxGetClassID(mxNonEmptyCols) != mxINT32_CLASS)
        mexErrMsgIdAndTxt(errId, "Invalid non empty indexes");
    int N_ACTIVE_CELL = max(mxNonEmptyRowsDim[0], mxNonEmptyRowsDim[1]);
    int *h_nonEmptyRows = (int*)mxGetPr(mxNonEmptyRows);
    int *h_nonEmptyCols = (int*)mxGetPr(mxNonEmptyCols);


    /* Lambda, added to the diagonals of Sigma */
    if (mxGetClassID(prhs[LAMBDA_IN]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Invalid input: lambda not double");
    float lambda = (float)mxGetScalar(prhs[LAMBDA_IN]);


    /* FEATURE_THRESHOLD */
    if (nrhs > 5 && mxGetClassID(prhs[FEATURE_THRESHOLD_IN]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Invalid input: lambda not double");
    if (nrhs > 5)
        FEATURE_THRESHOLD = (float)mxGetScalar(prhs[FEATURE_THRESHOLD_IN]);


    /* CG_TOLERANCE */
    if (nrhs > 6 && mxGetClassID(prhs[CG_TOLERANCE_IN]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Invalid input: CG_TOLERANCE not double");
    if (nrhs > 6)
        CG_TOLERANCE = (float)mxGetScalar(prhs[CG_TOLERANCE_IN]);


    /* CG_MAX_ITER */
    if (nrhs > 7 && mxGetClassID(prhs[CG_MAX_ITER_IN]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Invalid input: CG_MAX_ITER not double");
    if (nrhs > 7)
        CG_MAX_ITER = (int)mxGetScalar(prhs[CG_MAX_ITER_IN]);



    /* Check the Thread Size Parameters */
    if (( nrhs == 9)  && mxGetNumberOfElements(prhs[THREAD_SIZE_IN]) != 5)
        mexErrMsgIdAndTxt(errId, "CUDA Thread Size must be 4 integers : THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, THREAD_PER_BLOCK_2D_H, THREAD_PER_BLOCK_2D_W\nYou must choose size such that total thread will not be larger than MaxThreadsPerBlock");

    if ( nrhs == 9 ){
        const double* threadSize = (double *)mxGetData(prhs[THREAD_SIZE_IN]);
        THREAD_PER_BLOCK_H = (int)threadSize[0];
        THREAD_PER_BLOCK_W = (int)threadSize[1];
        THREAD_PER_BLOCK_D = (int)threadSize[2];
        THREAD_PER_BLOCK_2D_H = (int)threadSize[3];
        THREAD_PER_BLOCK_2D_W = (int)threadSize[4];
        if(debug) fprintf(stderr,"Thread size: H=%d, W=%d, D=%d, 2D=%d\n",
                                    THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, 
                                    THREAD_PER_BLOCK_2D_H, THREAD_PER_BLOCK_2D_W);
    }


    // cudaDeviceProp dev;
    // cudaGetDeviceProperties(&dev,0);
    // int success = checkDeviceProp(dev);
    


    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);
    /* Find number of cuda capable devices */
    // CUDA_SAFE_CALL(cudaGetDeviceCount(&N_GPU));
    // if(debug) fprintf(stderr, "CUDA-capable device count: %i\n", N_GPU);

    /* Setup Variables */
    int sigmaDim    = N_ACTIVE_CELL * FEATURE_DIM;
    /* Set block size and thread size */
    // dim3 threadBlock3D(THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D);
    // dim3 dataBlockGrid3D( iDivUp(WIDTH, threadBlock3D.x), 
    //                       iDivUp(HEIGHT, threadBlock3D.y), 
    //                       iDivUp(FEATURE_DIM, threadBlock3D.z));

    dim3 threadBlock2D( THREAD_PER_BLOCK_2D_W, THREAD_PER_BLOCK_2D_H);
    dim3 dataBlockGrid2D( iDivUp(sigmaDim, threadBlock2D.x), 
                          iDivUp(sigmaDim, threadBlock2D.y));


    thrust::device_vector<float> vec_d_Sigma(sigmaDim * sigmaDim);
    float* d_Sigma  = thrust::raw_pointer_cast(&vec_d_Sigma[0]);

    thrust::device_vector<int> vec_d_nonEmptyRows(h_nonEmptyRows, h_nonEmptyRows + N_ACTIVE_CELL);
    thrust::device_vector<int> vec_d_nonEmptyCols(h_nonEmptyCols, h_nonEmptyCols + N_ACTIVE_CELL);
    int* d_nonEmptyRows  = thrust::raw_pointer_cast(&vec_d_nonEmptyRows[0]);
    int* d_nonEmptyCols  = thrust::raw_pointer_cast(&vec_d_nonEmptyCols[0]);
    scrambleGammaToSigma<<<dataBlockGrid2D, threadBlock2D>>>( d_Sigma,
                                d_Gamma,
                                lambda,
                                d_nonEmptyRows, 
                                d_nonEmptyCols,
                                GammaDim, FEATURE_DIM, N_ACTIVE_CELL );
    

    /////////////////////////////////////////////
    ///    DEBUG
    /////////////////////////////////////////////
    if (debug){
        // thrust::host_vector<float> vec_h_Sigma = vec_d_Sigma;
        mwSize mwSigmaDim[2];
        mwSigmaDim[0] = sigmaDim; mwSigmaDim[1] = sigmaDim;
        plhs[SIGMA_OUT] = mxCreateNumericArray(2, mwSigmaDim, mxSINGLE_CLASS, mxREAL);
        float* h_Sigma = (float *)mxGetData(plhs[SIGMA_OUT]);
        cudaMemcpy(h_Sigma, d_Sigma, sigmaDim * sigmaDim * sizeof(float) ,cudaMemcpyDeviceToHost);
    }
    /////////////////////////////////////////////




    cublasDestroy(cublasHandle);
    mxGPUDestroyGPUArray(mxGamma);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    // cudaDeviceReset();
}
