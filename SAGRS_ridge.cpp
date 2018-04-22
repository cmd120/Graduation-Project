#include <math.h>
#include <stdlib.h>
#include "mex.h"
#include "mkl.h"
#include <random>
#include "comm.h"
#include <time.h>
#define DEBUG 1

/*
SAGRS_ridge(w,Xt,y,lambda,eta,d,g);
% w(p,1) - updated in place
% Xt(p,n) - real, can be sparse
% y(n,1) - {0,1}
% lambda - scalar regularization param
% d(p,1) - initial approximation of average gradient (should be sum of previous gradients)
% g(n,1) - previous derivatives of loss
% maxIter - scalar maximal iterations
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

int InnerLoopSingleDense(double *w, double *Xt, double *y, double lambda, double *d, double *g, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma);

int InnerLoopSingleSparse(double *w, double *Xt, mwIndex *ir, mwIndex *jc, double *y, double lambda, double *d, double *g, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma);

int InnerLoopBatchDense(double *w, double *Xt, double *y, double lambda, double *d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);

int InnerLoopBatchSparse(double *w, double *Xt, mwIndex *ir, mwIndex *jc, double *y, double lambda, double *d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);

// Calculate time taken by a request
struct timespec requestStart, requestEnd;
struct timespec cpuStart, cpuEnd;
int epochCounter;
double telapsed;
// test data
const mxArray *XtTest;
double *yTest;
double maxRunTime;
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
    double *w, *Xt, *y, lambda, a, b, gamma, *d, *g;

    if (nrhs != 16)
        mexErrMsgTxt("Function needs 16 arguments");

    /* Input */

    w = mxGetPr(prhs[0]);
    Xt = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    d = mxGetPr(prhs[4]);
    g = mxGetPr(prhs[5]);
    maxIter = (long)mxGetScalar(prhs[6]);
    batchSize = (int)mxGetScalar(prhs[7]);
    pass = (int)mxGetScalar(prhs[8]);
    a = mxGetScalar(prhs[9]);
    b = mxGetScalar(prhs[10]);
    gamma = mxGetScalar(prhs[11]);
    // @NOTE for printing error
    XtTest = prhs[12];
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
    nVars = mxGetM(prhs[1]);
    nSamples = mxGetN(prhs[1]);

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

    if (nVars != (int)mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != (int)mxGetM(prhs[2]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");
    if (nVars != (int)mxGetM(prhs[4]))
        mexErrMsgTxt("Xt and d must have the same number of rows");
    if (nSamples != (int)mxGetM(prhs[5]))
        mexErrMsgTxt("Xt and g must have the same number of columns");
    if (batchSize <= 0)
        mexErrMsgTxt("batchSize must be a positive integer");

    // sparse matrix
    if (mxIsSparse(prhs[1]))
    {
        sparse = 1;
        jc = mxGetJc(prhs[1]);
        ir = mxGetIr(prhs[1]);
    }

    int s, ret;
    for (s = 0; s < pass; s++)
    {
        if (batchSize == 1)
        {
            if (sparse)
            {
                ret = InnerLoopSingleSparse(w, Xt, ir, jc, y, lambda, d, g, maxIter, nSamples, nVars, s, a, b, gamma);
            }
            else
                ret = InnerLoopSingleDense(w, Xt, y, lambda, d, g, maxIter, nSamples, nVars, s, a, b, gamma);
        }
        else  // use mini-batch
        {
            if (sparse)
                ret = InnerLoopBatchSparse(w, Xt, ir, jc, y, lambda, d, g, maxIter, nSamples, nVars, batchSize, s, a, b, gamma);
            else
                ret = InnerLoopBatchDense(w, Xt, y, lambda, d, g, maxIter, nSamples, nVars, batchSize, s, a, b, gamma);
        }
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


int InnerLoopSingleDense(double *w, double *Xt, double *y, double lambda, double *d, double *g, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma)
{
    long i, idx, j;
    double innerProd, tmpDelta, eta;

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
        eta = a * pow(b+i+1, -gamma);
        idx = indices[i % nSamples];
#if NOISY
        std::normal_distribution<double> normalDist(/*mean=*/0.0, /*stddev=*/sqrt(eta*2/nSamples));
        noise = normalDist(genNoise);  // sample a noise
#endif

        /*  Compute derivative of loss \nabla f(w_{idx}) */
        innerProd = cblas_ddot(nVars, Xt + nVars * idx, 1, w, 1);
        tmpDelta = PartialGradient(innerProd, y[idx]);

        cblas_daxpby(nVars, -eta/nSamples, d, 1, 1 - eta * lambda, w, 1);
        for(j = 0; j < nVars; j++)  // add noise
            w[j] += noise;

        cblas_daxpy(nVars, -eta * (tmpDelta - g[idx]) / nSamples, Xt + nVars * idx, 1, w, 1);  // @NOTE biased estimator
        /* Update direction */
        cblas_daxpy(nVars, tmpDelta - g[idx], Xt + nVars * idx, 1, d, 1);
        g[idx] = tmpDelta;

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
    }
    return 0;
}


int InnerLoopSingleSparse(double *w, double *Xt, mwIndex *ir, mwIndex *jc, double *y, double lambda, double *d, double *g, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma)
{
    int *lastVisited;
    long i, idx, j;
    double innerProd, tmpGrad, c = 1, tmpFactor, *cumSum, *cumNoise, eta;

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
        eta = a * pow(b+i+1, -gamma);
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
                    w[ir[j]] += -d[ir[j]] * cumSum[i-1] + cumNoise[i-1];
                else // if lastVisited[ir[j]] > 0
                    w[ir[j]] += -d[ir[j]] * (cumSum[i-1] - cumSum[lastVisited[ir[j]]-1]) + cumNoise[i-1] - cumNoise[lastVisited[ir[j]]-1];
                lastVisited[ir[j]] = i;
            }
        }

        /* Step 2:  Compute derivative of loss \nabla f(w_{i}) */
        innerProd = 0;
        for(j = jc[idx]; j < (long)jc[idx+1]; j++)
        {
            innerProd += w[ir[j]] * Xt[j];
        }
        innerProd *= c;  // rescale
        tmpGrad = PartialGradient(innerProd, y[idx]);

        // update cumSum
        c *= 1-eta*lambda;
        tmpFactor = eta/c/nSamples;

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
        tmpFactor = eta/c/nSamples * (tmpGrad - g[idx]);  // @NOTE biased estimator
        cblas_daxpyi(jc[idx+1] - jc[idx], -tmpFactor, Xt + jc[idx], (int *)(ir + jc[idx]), w);
        // @NOTE (int *) here is 64bit because mwIndex is 64bit, and we have to link libmkl_intel_ilp64.a for 64bit integer

        /* Step 4: update d and g[idx] */
        for(j = jc[idx]; j < (long)jc[idx+1]; j++)
            d[ir[j]] += Xt[j]*(tmpGrad - g[idx]);
        g[idx] = tmpGrad;

        // Re-normalize the parameter vector if it has gone numerically crazy
        if(((i+1) % maxIter == maxIter*epochCounter/PRINT_FREQ) || c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100))
        {
            for(j = 0; j < nVars; j++)
            {
                if (lastVisited[j] == 0)
                    w[j] += -d[j] * cumSum[i] + cumNoise[i];
                else
                    w[j] += -d[j] * (cumSum[i]-cumSum[lastVisited[j]-1]) + cumNoise[i] - cumNoise[lastVisited[j]-1];
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
            w[j] += -d[j] * cumSum[maxIter-1] + cumNoise[maxIter-1];
        else
            w[j] += -d[j] * (cumSum[maxIter-1] - cumSum[lastVisited[j]-1]) + cumNoise[maxIter-1] - cumNoise[lastVisited[j]-1];
    }
    cblas_dscal(nVars, c, w, 1);
    mxFree(lastVisited);
    mxFree(cumSum);
    mxFree(cumNoise);
    return 0;
}

int InnerLoopBatchDense(double *w, double *Xt, double *y, double lambda, double *d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma)
{
    long i, idx, j;
    double innerProd, *gradBuffer, eta;
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
        eta = a * pow(b+i+1, -gamma);
#if NOISY
        std::normal_distribution<double> normalDist(/*mean=*/0.0, /*stddev=*/sqrt(eta*2/nSamples));
        noise = normalDist(genNoise);  // sample a noise
#endif

        /*  Compute derivative of loss \nabla f(w_{i}) */
        for (k = 0; k < batchSize; k++)
        {
            idx = indices[(i * batchSize + k) % nSamples];
            sampleBuffer[k] = idx;
            innerProd = cblas_ddot(nVars, Xt + nVars * idx, 1, w, 1);
            gradBuffer[k] = PartialGradient(innerProd, y[idx]);
        }

        cblas_daxpby(nVars, -eta/nSamples, d, 1, 1 - eta * lambda, w, 1);
        for(j = 0; j < nVars; j++)  // add noise
            w[j] += noise;

        for (k = 0; k < batchSize; k++)
        {
            idx = sampleBuffer[k];
            cblas_daxpy(nVars, -eta * (gradBuffer[k] - g[idx]) / nSamples, Xt + nVars * idx, 1, w, 1);  // @NOTE biased estimator
        }
        /* update direction */
        for (k = 0; k < batchSize; k++)
        {
            idx = sampleBuffer[k];
            cblas_daxpy(nVars, gradBuffer[k] - g[idx], Xt + nVars * idx, 1, d, 1);
            g[idx] = gradBuffer[k];
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

int InnerLoopBatchSparse(double *w, double *Xt, mwIndex *ir, mwIndex *jc, double *y, double lambda, double *d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma)
{
    int *lastVisited, *sampleBuffer, k;
    long i, idx, j;
    double innerProd, c = 1, tmpFactor, *cumSum, *cumNoise, *gradBuffer, eta;

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
        eta = a * pow(b+i+1, -gamma);
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
                        w[ir[j]] += -d[ir[j]] * cumSum[i-1] + cumNoise[i-1];
                    else if (lastVisited[ir[j]] != i) // if lastVisited[ir[j]] > 0 && != i
                        w[ir[j]] += -d[ir[j]] * (cumSum[i-1] - cumSum[lastVisited[ir[j]]-1]) + cumNoise[i-1] - cumNoise[lastVisited[ir[j]]-1];
                    lastVisited[ir[j]] = i;
                }
            }
        }

        /* Step 2:  Compute derivative of loss \nabla f(w_{i}) */
        for (k = 0; k < batchSize; k++)
        {
            idx = sampleBuffer[k];
            innerProd = 0;
            for(j = jc[idx]; j < (long)jc[idx+1]; j++)
            {
                innerProd += w[ir[j]] * Xt[j];
            }
            innerProd *= c;  // rescale
            gradBuffer[k] = PartialGradient(innerProd, y[idx]);
        }

        // update cumSum
        c *= 1-eta*lambda;
        tmpFactor = eta/c/nSamples;

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
            tmpFactor = eta/c/nSamples * (gradBuffer[k] - g[idx]);  // @NOTE biased estimator
            cblas_daxpyi(jc[idx+1] - jc[idx], -tmpFactor, Xt + jc[idx], (int *)(ir + jc[idx]), w);
            // @NOTE (int *) here is 64bit because mwIndex is 64bit, and we have to link libmkl_intel_ilp64.a for 64bit integer
        }

        /* Step 4: update d and g[idx] */
        for (k = 0; k < batchSize; k++)
        {
            idx = sampleBuffer[k];
            for(j = jc[idx]; j < (long)jc[idx+1]; j++)
                d[ir[j]] += Xt[j]*(gradBuffer[k] - g[idx]);
            g[idx] = gradBuffer[k];
        }

        // Re-normalize the parameter vector if it has gone numerically crazy
        if(((i+1) % maxIter == maxIter*epochCounter/PRINT_FREQ) || c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100))
        {
            for(j = 0; j < nVars; j++)
            {
                if (lastVisited[j] == 0)
                    w[j] += -d[j] * cumSum[i] + cumNoise[i];
                else
                    w[j] += -d[j] * (cumSum[i]-cumSum[lastVisited[j]-1]) + cumNoise[i] - cumNoise[lastVisited[j]-1];
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
            w[j] += -d[j] * cumSum[maxIter-1] + cumNoise[maxIter-1];
        else
            w[j] += -d[j] * (cumSum[maxIter-1] - cumSum[lastVisited[j]-1]) + cumNoise[maxIter-1] - cumNoise[lastVisited[j]-1];
    }
    cblas_dscal(nVars, c, w, 1);
    mxFree(lastVisited);
    mxFree(cumSum);
    mxFree(cumNoise);
    mxFree(gradBuffer);
    mxFree(sampleBuffer);
    return 0;
}
