#include <math.h>
#include <stdlib.h>
#include "mex.h"
#include "mkl.h"
#include <random>
#include "comm.h"
#include <time.h>
#define DEBUG 1
#define DEBUG_SHUFFLE 0

/*
SVRGRS_ridge(w,Xt,y,lambda,eta,d,g);
% w(p,1) - the iterate, updated in place
% wtilde(p,1) - snapshot, updated in place
% G(p,1) - full gradient of snapshot, updated in place
% Xt(p,n) - real, can be sparse
% y(n,1) - {0,1}
% lambda - scalar regularization param
% maxIter - maximal iterations of inner loop
% batchSize - mini-batch size m
% pass - used to determine #iteration
% a - used to determine step sizes
% b - used to determine step sizes
% gamma - used to determine step sizes
% XtTest
% ytest
% maxRunTime
% filename - saving results
*/

inline double PartialGradient(double innerProd, double y)
{
    return innerProd - y;
}

int InnerLoopSingleDense(double *w, double *wtilde, double *G, double *Xt, double *y, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma);
int InnerLoopSingleSparse(double *w, double *wtilde, double *G, double *Xt, mwIndex *ir, mwIndex *jc, double *y, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma);
int InnerLoopBatchDense(double *w, double *wtilde, double *G, double *Xt, double *y, double lambda, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int InnerLoopBatchSparse(double *w, double *wtilde, double *G, double *Xt, mwIndex *ir, mwIndex *jc, double *y, double lambda, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);

// Calculate time taken by a request
struct timespec requestStart, requestEnd;
struct timespec cpuStart, cpuEnd;
int epochCounter;
double telapsed;
// test data
const mxArray *XtTest;
double *yTest;
double maxRunTime;
const mxArray *XtArr;
// file and buffer
FILE *fp;
char filename[FILE_NAME_LENGTH];
int *indices;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // timing
    clock_gettime(CLOCK_MONOTONIC_RAW, &requestStart);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpuStart);
    srand(time(NULL));
    /* Variables */
    int nSamples, nVars, batchSize;
    long maxIter;
    int sparse = 0, pass;
    mwIndex *jc, *ir;
    double *w, *wtilde, *G, *Xt, *y, lambda, a, b, gamma;

    if (nrhs != 16)
        mexErrMsgTxt("Function needs 16 arguments");

    /* Input */

    w = mxGetPr(prhs[0]);
    wtilde = mxGetPr(prhs[1]);
    G = mxGetPr(prhs[2]);
    Xt = mxGetPr(prhs[3]);
    y = mxGetPr(prhs[4]);  // we do not need y
    lambda = mxGetScalar(prhs[5]);
    maxIter = (long)mxGetScalar(prhs[6]);
    batchSize = (int)mxGetScalar(prhs[7]);
    pass = (int)mxGetScalar(prhs[8]);
    a = mxGetScalar(prhs[9]);
    b = mxGetScalar(prhs[10]);
    gamma = mxGetScalar(prhs[11]);
    // @NOTE for printing error
    XtTest = prhs[12];
    XtArr = prhs[3];
    yTest = mxGetPr(prhs[13]);  // we do not need y
    maxRunTime = mxGetScalar(prhs[14]);
    mxGetString(prhs[15], filename, FILE_NAME_LENGTH);
    fp = fopen(filename, "a");
    if (NULL == fp)
    {
        mexErrMsgTxt("Cannot open file to write results");
    }
    // @NOTE compute initial error
    epochCounter = 0;
    RidgeError(w, XtTest, yTest, 0, 0, fp);
    epochCounter = (epochCounter + 1) % PRINT_FREQ;

    /* Compute Sizes */
    nVars = mxGetM(prhs[3]);
    nSamples = mxGetN(prhs[3]);

    // initialize indices
    indices = (int *)mxCalloc(nSamples,sizeof(int));
    if (NULL == indices)
    {
        mexErrMsgTxt("Cannot open allocate indices buffer");
    }
    int i;
    for (i = 0; i < nSamples; i++)
    {
        indices[i] = i;
    }
    Shuffle(indices, nSamples);


    if (nVars != (int)mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != (int)mxGetM(prhs[4]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");
    if (batchSize <= 0)
        mexErrMsgTxt("batchSize must be a positive integer");

    // sparse matrix
    if (mxIsSparse(prhs[3]))
    {
        sparse = 1;
        jc = mxGetJc(prhs[3]);
        ir = mxGetIr(prhs[3]);
    }



    int s, ret;
    for (s = 0; s < pass; s++)
    {
        RidgeGradient(wtilde, XtArr, y, G);
        if (batchSize == 1)
        {
            if (sparse)
                ret = InnerLoopSingleSparse(w, wtilde, G, Xt, ir, jc, y, lambda, maxIter, nSamples, nVars, s, a, b, gamma);
            else
                ret = InnerLoopSingleDense(w, wtilde, G, Xt, y, lambda, maxIter, nSamples, nVars, s, a, b, gamma);
        }
        else  // use mini-batch
        {
            if (sparse)
                ret = InnerLoopBatchSparse(w, wtilde, G, Xt, ir, jc, y, lambda, maxIter, nSamples, nVars, batchSize, s, a, b, gamma);
            else
                ret = InnerLoopBatchDense(w, wtilde, G, Xt, y, lambda, maxIter, nSamples, nVars, batchSize, s, a, b, gamma);
        }
        cblas_dcopy(nVars, w, 1, wtilde, 1);
        if (0 != ret)
            break;
    }

    // @NOTE caculate elapsed total time and CPU time
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpuEnd);
    double accumCPU = ( cpuEnd.tv_sec - cpuStart.tv_sec )
      + ( cpuEnd.tv_nsec - cpuStart.tv_nsec )
      / BILLION;
    clock_gettime(CLOCK_MONOTONIC_RAW, &requestEnd);
    double accum = ( requestEnd.tv_sec - requestStart.tv_sec )
      + ( requestEnd.tv_nsec - requestStart.tv_nsec )
      / BILLION;
    mexPrintf( "total wall time: %lf, total CPU time: %lf\n", accum, accumCPU);

    fclose(fp);
    mxFree(indices);
    return;
}

int InnerLoopSingleDense(double *w, double *wtilde, double *G, double *Xt, double *y, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma)
{
    long i, idx, j;
    double innerProdI, innerProdZ, tmpDelta, eta;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 genNoise(rd()), genSample(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> uniformDist(0, nSamples-1);
    double noise = 0; // noise ~ N(0, eta*2/n)

    for (i = 0; i < maxIter; i++)
    {
        if (0 == i % nSamples)
        {
            Shuffle(indices, nSamples);
        }
        eta = a * pow(b + pass*1.0*maxIter+i+1, -gamma);
        idx = indices[i % nSamples];
#if DEBUG_SHUFFLE
        mexPrintf("%d, ", idx);
#endif

#if NOISY
        std::normal_distribution<double> normalDist(/*mean=*/0.0, /*stddev=*/sqrt(eta*2/nSamples));
        noise = normalDist(genNoise);  // sample a noise
#endif

        /*  Compute derivative of loss \nabla f(w_{i}) */
        innerProdI = 0;
        innerProdZ = 0;
        for(j = 0; j < nVars; j++)
        {
            innerProdI += w[j] * Xt[j + nVars * idx];
            innerProdZ += wtilde[j] * Xt[j + nVars * idx];
        }
        tmpDelta = PartialGradient(innerProdI, 0) - PartialGradient(innerProdZ, 0);

        cblas_daxpby(nVars, -eta, G, 1, 1 - eta * lambda, w, 1);
        for(j = 0; j < nVars; j++)  // add noise
            w[j] += noise;

        cblas_daxpy(nVars, -eta * tmpDelta, Xt + nVars * idx, 1, w, 1);

        // @NOTE compute error
        if ((i+1) % maxIter == maxIter*epochCounter/PRINT_FREQ)// print test error
        {
            clock_gettime(CLOCK_MONOTONIC_RAW, &requestEnd);
            telapsed = ( requestEnd.tv_sec - requestStart.tv_sec ) + ( requestEnd.tv_nsec - requestStart.tv_nsec ) / BILLION;

            RidgeError(w, XtTest, yTest, pass + (i+1)*1.0/maxIter, telapsed, fp);
            epochCounter = (epochCounter + 1) % PRINT_FREQ;
            if (telapsed >= maxRunTime)
            {
                return 1;
            }
        }

#if DEBUG_SHUFFLE
        if ((i+1) % nSamples == 0)
        {
            mexPrintf("\n");
        }
#endif
    }
    return 0;
}


int InnerLoopSingleSparse(double *w, double *wtilde, double *G, double *Xt, mwIndex *ir, mwIndex *jc, double *y, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma)
{
    int *lastVisited;
    long i, idx, j;
    double innerProdI, innerProdZ, tmpDelta, c = 1, tmpFactor, *cumSum, *cumNoise, eta;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 genNoise(rd()), genSample(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> uniformDist(0, nSamples-1);
    double noise = 0; // noise ~ N(0, eta*2/n)

    /* Allocate memory needed for lazy updates */
    lastVisited = (int *)mxCalloc(nVars,sizeof(int));  // O(d)
    cumSum = (double *)mxCalloc(maxIter,sizeof(double));  // O(T)
    cumNoise = (double *)mxCalloc(maxIter,sizeof(double));  // O(T)

    for (i = 0; i < maxIter; i++)
    {
        if (0 == i % nSamples)
        {
            Shuffle(indices, nSamples);
        }
        eta = a * pow(b + pass*1.0*maxIter+i+1, -gamma);
        idx = indices[i % nSamples];
#if NOISY
        std::normal_distribution<double> normalDist(/*mean=*/0.0, /*stddev=*/sqrt(eta*2/nSamples));
        noise = normalDist(genNoise);  // sample a noise
#endif
        /* Step 1: Compute current values of needed parameters w_{i} */
        if (i > 0)
        {
            for(j = jc[idx]; j < (long)jc[idx+1]; j++)
            {
                if (lastVisited[ir[j]] == 0)  // or we can let lastVisited[-1] = 0
                    w[ir[j]] += -G[ir[j]] * cumSum[i-1] + cumNoise[i-1];
                else // if lastVisited[ir[j]] > 0
                    w[ir[j]] += -G[ir[j]] * (cumSum[i-1] - cumSum[lastVisited[ir[j]]-1]) + cumNoise[i-1] - cumNoise[lastVisited[ir[j]]-1];
                lastVisited[ir[j]] = i;
            }
        }

        /* Step 2:  Compute derivative of loss \nabla f(w_{i}) */
        innerProdI = 0;
        innerProdZ = 0;
        for(j = jc[idx]; j < (long)jc[idx+1]; j++)
        {
            innerProdI += w[ir[j]] * Xt[j];
            innerProdZ += wtilde[ir[j]] * Xt[j];
        }
        innerProdI *= c;  // rescale
        tmpDelta = PartialGradient(innerProdI, 0) - PartialGradient(innerProdZ, 0);

        // update cumSum
        c *= 1-eta*lambda;
        tmpFactor = eta/c;

        if (i == 0)
        {
            cumSum[0] = tmpFactor;
            cumNoise[0] = noise/c;
        }
        else
        {
            cumSum[i] = cumSum[i-1] + tmpFactor;
            cumNoise[i] = cumNoise[i-1] + noise/c;
        }
        tmpFactor = eta/c * tmpDelta; // tmpFactor is used for next if-else

        /* Step 3: approximate w_{i+1} */
        cblas_daxpyi(jc[idx+1] - jc[idx], -tmpFactor, Xt + jc[idx], (int *)(ir + jc[idx]), w);
        // @NOTE (int *) here is 64bit because mwIndex is 64bit, and we have to link libmkl_intel_ilp64.a for 64bit integer

        // Re-normalize the parameter vector if it has gone numerically crazy
        if(((i+1) % maxIter == maxIter*epochCounter/PRINT_FREQ) || c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100))
        {
            for(j = 0; j < nVars; j++)
            {
                if (lastVisited[j] == 0)
                    w[j] += -G[j] * cumSum[i] + cumNoise[i];
                else
                    w[j] += -G[j] * (cumSum[i]-cumSum[lastVisited[j]-1]) + cumNoise[i] - cumNoise[lastVisited[j]-1];
                lastVisited[j] = i+1;
            }
            cumSum[i] = 0;
            cumNoise[i] = 0;
            cblas_dscal(nVars, c, w, 1);
            c = 1;

            // @NOTE compute error
            if ((i+1) % maxIter == maxIter*epochCounter/PRINT_FREQ)// print test error
            {
                clock_gettime(CLOCK_MONOTONIC_RAW, &requestEnd);
                telapsed = ( requestEnd.tv_sec - requestStart.tv_sec ) + ( requestEnd.tv_nsec - requestStart.tv_nsec ) / BILLION;

                RidgeError(w, XtTest, yTest, pass + (i+1)*1.0/maxIter, telapsed, fp);
                epochCounter = (epochCounter + 1) % PRINT_FREQ;
                if (telapsed >= maxRunTime)
                {
                    mxFree(lastVisited);
                    mxFree(cumSum);
                    mxFree(cumNoise);
                    return 1;
                }
            }
        }

    }

    // at last, correct the iterate once more
    for(j = 0; j < nVars; j++)
    {
        if (lastVisited[j] == 0)
            w[j] += -G[j] * cumSum[maxIter-1] + cumNoise[maxIter-1];
        else
            w[j] += -G[j] * (cumSum[maxIter-1] - cumSum[lastVisited[j]-1]) + cumNoise[maxIter-1] - cumNoise[lastVisited[j]-1];
    }
    cblas_dscal(nVars, c, w, 1);
    mxFree(lastVisited);
    mxFree(cumSum);
    mxFree(cumNoise);
    return 0;
}

int InnerLoopBatchDense(double *w, double *wtilde, double *G, double *Xt, double *y, double lambda, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma)
{
    long i, idx, j;
    double innerProdI, innerProdZ, *gradBuffer, eta;
    int *sampleBuffer, k;

    gradBuffer = (double *)mxCalloc(batchSize, sizeof(double));  // O(m)
    sampleBuffer = (int *)mxCalloc(batchSize, sizeof(int));  // O(m)

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 genNoise(rd()), genSample(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> uniformDist(0, nSamples-1);
    double noise = 0; // noise ~ N(0, eta*2/n)

    for (i = 0; i < maxIter; i++)
    {
        if (0 == i % nSamples)
        {
            Shuffle(indices, nSamples);
        }
        eta = a * pow(b + pass*1.0*maxIter+i+1, -gamma);
#if NOISY
        std::normal_distribution<double> normalDist(/*mean=*/0.0, /*stddev=*/sqrt(eta*2/nSamples));
        noise = normalDist(genNoise);  // sample a noise
#endif

        /*  Compute derivative of loss \nabla f(w_{i}) */
        for (k = 0; k < batchSize; k++)
        {
            idx = indices[(i * batchSize + k) % nSamples];
            sampleBuffer[k] = idx;
            innerProdI = cblas_ddot(nVars, Xt + nVars * idx, 1, w, 1);
            innerProdZ = cblas_ddot(nVars, Xt + nVars * idx, 1, wtilde, 1);
            gradBuffer[k] = PartialGradient(innerProdI, 0) - PartialGradient(innerProdZ, 0);
        }

        cblas_daxpby(nVars, -eta, G, 1, 1 - eta * lambda, w, 1);
        for(j = 0; j < nVars; j++)  // add noise
            w[j] += noise;

        for (k = 0; k < batchSize; k++)
        {
            idx = sampleBuffer[k];
            cblas_daxpy(nVars, -eta * gradBuffer[k] / batchSize, Xt + nVars * idx, 1, w, 1);
        }

        // @NOTE compute error
        if ((i+1) % maxIter == maxIter*epochCounter/PRINT_FREQ)// print test error
        {
            clock_gettime(CLOCK_MONOTONIC_RAW, &requestEnd);
            telapsed = ( requestEnd.tv_sec - requestStart.tv_sec ) + ( requestEnd.tv_nsec - requestStart.tv_nsec ) / BILLION;

            RidgeError(w, XtTest, yTest, pass + (i+1)*1.0/maxIter, telapsed, fp);
            epochCounter = (epochCounter + 1) % PRINT_FREQ;
            if (telapsed >= maxRunTime)
            {
                mxFree(gradBuffer);
                mxFree(sampleBuffer);
                return 1;
            }
        }
    }
    mxFree(gradBuffer);
    mxFree(sampleBuffer);
    return 0;
}

int InnerLoopBatchSparse(double *w, double *wtilde, double *G, double *Xt, mwIndex *ir, mwIndex *jc, double *y, double lambda, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma)
{
    int *lastVisited, *sampleBuffer, k;
    long i, idx, j;
    double innerProdI, innerProdZ, c = 1, tmpFactor, *cumSum, *cumNoise, *gradBuffer, eta;

    gradBuffer = (double *)mxCalloc(batchSize, sizeof(double));  // O(m)
    sampleBuffer = (int *)mxCalloc(batchSize, sizeof(int));  // O(m)

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 genNoise(rd()), genSample(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> uniformDist(0, nSamples-1);
    double noise = 0; // noise ~ N(0, eta*2/n)

    /* Allocate memory needed for lazy updates */
    lastVisited = (int *)mxCalloc(nVars,sizeof(int));  // O(d)
    cumSum = (double *)mxCalloc(maxIter,sizeof(double));  // O(T)
    cumNoise = (double *)mxCalloc(maxIter,sizeof(double));  // O(T)

    for (i = 0; i < maxIter; i++)
    {
        if (0 == i % nSamples)
        {
            Shuffle(indices, nSamples);
        }
        eta = a * pow(b + pass*1.0*maxIter+i+1, -gamma);
#if NOISY
        std::normal_distribution<double> normalDist(/*mean=*/0.0, /*stddev=*/sqrt(eta*2/nSamples));
        noise = normalDist(genNoise);  // sample a noise
#endif
        for (k = 0; k < batchSize; k++)
            sampleBuffer[k] = indices[(i * batchSize + k) % nSamples];

        /* Step 1: Compute current values of needed parameters w_{i} */
        if (i > 0)
        {
            for (k = 0; k < batchSize; k++)
            {
                idx = sampleBuffer[k];
                for(j = jc[idx]; j < (long)jc[idx+1]; j++)
                {
                    if (lastVisited[ir[j]] == 0) // or we can let lastVisited[-1] = 0
                        w[ir[j]] += -G[ir[j]] * cumSum[i-1] + cumNoise[i-1];
                    else if (lastVisited[ir[j]] != i) // if lastVisited[ir[j]] > 0 && != i
                        w[ir[j]] += -G[ir[j]] * (cumSum[i-1] - cumSum[lastVisited[ir[j]]-1]) + cumNoise[i-1] - cumNoise[lastVisited[ir[j]]-1];
                    lastVisited[ir[j]] = i;
                }
            }
        }

        /* Step 2:  Compute derivative of loss \nabla f(w_{i}) */
        for (k = 0; k < batchSize; k++)
        {
            idx = sampleBuffer[k];
            innerProdI = 0;
            innerProdZ = 0;
            for(j = jc[idx]; j < (long)jc[idx+1]; j++)
            {
                innerProdI += w[ir[j]] * Xt[j];
                innerProdZ += wtilde[ir[j]] * Xt[j];
            }
            innerProdI *= c;  // rescale
            gradBuffer[k] = PartialGradient(innerProdI, 0) - PartialGradient(innerProdZ, 0);
        }

        // update cumSum
        c *= 1-eta*lambda;
        tmpFactor = eta/c;

        if (i == 0)
        {
            cumSum[0] = tmpFactor;
            cumNoise[0] = noise/c;
        }
        else
        {
            cumSum[i] = cumSum[i-1] + tmpFactor;
            cumNoise[i] = cumNoise[i-1] + noise/c;
        }

        /* Step 3: approximate w_{i+1} */
        for (k = 0; k < batchSize; k++)
        {
            idx = sampleBuffer[k];
            tmpFactor = eta/c/batchSize * gradBuffer[k];
            cblas_daxpyi(jc[idx+1] - jc[idx], -tmpFactor, Xt + jc[idx], (int *)(ir + jc[idx]), w);
            // @NOTE (int *) here is 64bit because mwIndex is 64bit, and we have to link libmkl_intel_ilp64.a for 64bit integer
        }

        // Re-normalize the parameter vector if it has gone numerically crazy
        if(((i+1) % maxIter == maxIter*epochCounter/PRINT_FREQ) || c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100))
        {
            for(j = 0; j < nVars; j++)
            {
                if (lastVisited[j] == 0)
                    w[j] += -G[j] * cumSum[i] + cumNoise[i];
                else
                    w[j] += -G[j] * (cumSum[i]-cumSum[lastVisited[j]-1]) + cumNoise[i] - cumNoise[lastVisited[j]-1];
                lastVisited[j] = i+1;
            }
            cumSum[i] = 0;
            cumNoise[i] = 0;
            cblas_dscal(nVars, c, w, 1);
            c = 1;

            // @NOTE compute error
            if ((i+1) % maxIter == maxIter*epochCounter/PRINT_FREQ)// print test error
            {
                clock_gettime(CLOCK_MONOTONIC_RAW, &requestEnd);
                telapsed = ( requestEnd.tv_sec - requestStart.tv_sec ) + ( requestEnd.tv_nsec - requestStart.tv_nsec ) / BILLION;

                RidgeError(w, XtTest, yTest, pass + (i+1)*1.0/maxIter, telapsed, fp);
                epochCounter = (epochCounter + 1) % PRINT_FREQ;
                if (telapsed >= maxRunTime)
                {
                    mxFree(lastVisited);
                    mxFree(cumSum);
                    mxFree(cumNoise);
                    mxFree(gradBuffer);
                    mxFree(sampleBuffer);
                    return 1;
                }
            }
        }
    }

    // at last, correct the iterate once more
    for(j = 0; j < nVars; j++)
    {
        if (lastVisited[j] == 0)
            w[j] += -G[j] * cumSum[maxIter-1] + cumNoise[maxIter-1];
        else
            w[j] += -G[j] * (cumSum[maxIter-1] - cumSum[lastVisited[j]-1]) + cumNoise[maxIter-1] - cumNoise[lastVisited[j]-1];
    }
    cblas_dscal(nVars, c, w, 1);
    mxFree(lastVisited);
    mxFree(cumSum);
    mxFree(cumNoise);
    mxFree(gradBuffer);
    mxFree(sampleBuffer);
    return 0;
}
