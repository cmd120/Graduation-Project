#include "include/SGD.h"

/*
SGD_ridge(w,Xt,y,lambda,eta,d,g);
% w(p,1) - the iterate, updated in place
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

void SGD_ridge(VectorXd &w, const MatrixXd &Xt, VectorXd y, const MatrixXd &XtTest, \
    const VectorXd &yTest, VectorXd d, VectorXd g, string filename, double lambda, double eta, \
    int maxIter, int batchSize, int pass, int a, int b, int gamma,  int maxRunTime) {
    
    int nVars, nSamples, flag;
    int epochCounter = 0;
    nVars = Xt.rows();
    nSamples = Xt.cols();
    FILE *fp = fopen(filename.c_str(), "a");
    if (fp == NULL) {
        cout << "Cannot write results to file: " << filename << endl;
    }
    RidgeError(w, XtTest, yTest, 0, 0, fp);
    epochCounter = (epochCounter + 1) % PRINT_FREQ;
    //为什么ret会在循环内部不断更新
    for (int i = 0; i < pass; i++) {
        flag = batchSize?InnerLoopBatchDense(w, Xt, y, d, g, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, batchSize):\
                            InnerLoopSingleDense(w, Xt, y, d, g, lambda, maxIter, nSamples, nVars, pass, a, b, gamma);
        if (flag) {
            break;
        }
    }
    fclose(fp);
}



int InnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd y, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma)
{
    long i, idx, j;
    double innerProd = 0 , tmpDelta, eta;
    Noise idxSample(0,nSamples-1);
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    for (i = 0; i < maxIter; i++) {
        eta = a * pow(b + i + 1, -gamma);
        idx = idxSample.gen();
        innerProd += w(j)*Xt.col(idx)(j);
        tmpDelta = RidgePartialGradient(innerProd, y(idx));
        w = -eta*tmpDelta*Xt.col(idx)+(1-eta*lambda)*w;
        w = NOISY?w.array()+noise.gen():w;
        
        // //compute error
        // if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) {
        //  RidgeError(w, XtTest, yTest, pass + (i + 1)*1.0 / maxIter, telapsed, fp);
        //  epochCounter = (epochCounter + 1) % PRINT_FREQ;
        //  if (telapsed >= maxRunTime) {
        //      return 1;
        //  }
        // }
    }
    return 0;
}

int InnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd y, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize)
{
    long i, idx, j, k;
    double innerProd, eta;

    VectorXd gradBuffer(batchSize);
    int* sampleBuffer = new int[batchSize];

    Noise idxSample(0,nSamples-1);
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    
    for (i = 0; i < maxIter;i++) {
        eta = a * pow(b + i + 1, -gamma);
        
        for (k = 0; k < batchSize; k++) {
            idx = idxSample.gen();
            sampleBuffer[k] = idx;
            innerProd = Xt.col(idx).dot(w);
            gradBuffer(k) = RidgePartialGradient(innerProd, y(idx));
        }

        w = (1-eta*lambda) * w;
        
        w = NOISY?w.array()+noise.gen():w;

        for (k = 0; k < batchSize; k++) {
            idx = sampleBuffer[k];
            w += (-eta * gradBuffer(k)  / batchSize)*Xt.col(idx);
        }
        // if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) {
        //  ...
        //      RidgeError(w, XtTest, yTest, pass + (i + 1)*1.0 / maxIter, telapsed, fp);
        //  epochCounter = (epochCounter + 1) % PRINT_FREQ;
        //  if (telapsed >= maxRunTime) {
        //      return 1;
        //  }
        // }
    }
    delete[] sampleBuffer;
    return 0;
}