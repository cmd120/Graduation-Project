#include "include/SVRG.h"

/*
SVRG_logistic(w,Xt,y,lambda,eta,wtilde,G);
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
int epochCounter;
FILE *fp;
auto startTime = Clock::now();

void SVRG_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd y, const MatrixXd &XtTest, \
     VectorXd &yTest, VectorXd wtilde, VectorXd G, string filename, double lambda, double eta, \
    int maxIter, int batchSize, int pass, int a, int b, int gamma,  int maxRunTime) {
    
    startTime = Clock::now();

    int nVars, nSamples, flag;
    int epochCounter = 0;
    nVars = Xt.rows();
    nSamples = Xt.cols();
    FILE *fp = fopen(filename.c_str(), "a");
    if (fp == NULL) {
        cout << "Cannot write results to file: " << filename << endl;
    }
    epochCounter = 0;
    LogisticError(w, XtTest, yTest, 0, 0, fp);
    epochCounter = (epochCounter + 1) % PRINT_FREQ;
    //为什么ret会在循环内部不断更新
    for (int i = 0; i < pass; i++) {
        flag = batchSize?InnerLoopBatchDense(w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, batchSize, maxRunTime):\
                            InnerLoopSingleDense(w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);
        if (flag) {
            break;
        }
    }
    fclose(fp);

    auto endTime = Clock::now();
    cout << "duration: " << chrono::duration_cast<chrono::nanoseconds>(endTime-startTime).count()/BILLION << endl;

    return;
}


int InnerLoopSingleDense(VectorXd &w,const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &wtilde, VectorXd &G, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime)
{
    long i, idx, j;
    double innerProdI = 0, innerProdZ=0, tmpDelta, eta;
    Noise idxSample(0,nSamples-1);
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    for (i = 0; i < maxIter; i++) {
        eta = a * pow(b + pass*1.0*maxIter +i + 1, -gamma);
        idx = idxSample.gen();
        for(j=0;j<nVars;++i){
            innerProdI += w(j) * Xt.col(idx)(j);
            innerProdZ += wtilde[j] * Xt.col(idx)(j);
        }

        tmpDelta = LogisticPartialGradient(innerProdI,0)-LogisticPartialGradient(innerProdZ,0);
        
        w = -eta*G+(1-eta*lambda)*w;
        w = NOISY?w.array()+noise.gen():w;
        w = w + (-eta) * tmpDelta * Xt.col(idx);

        //compute error
        if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) {
            auto endTime = Clock::now();
            double telapsed = chrono::duration_cast<chrono::nanoseconds>(endTime-startTime).count()/BILLION;
            LogisticError(w, XtTest, yTest, pass + (i + 1)*1.0 / maxIter, telapsed, fp);
            epochCounter = (epochCounter + 1) % PRINT_FREQ;
            if (telapsed >= maxRunTime) {
                return 1;
            }
        }
    }
    return 0;
}

int InnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &wtilde, VectorXd &G, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize, int maxRunTime)
{
    long i, idx, j, k;
    double innerProdI=0,innerProdZ=0, eta;

    VectorXd gradBuffer(batchSize);
    int* sampleBuffer = new int[batchSize];

    Noise idxSample(0, nSamples-1);
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    for (i = 0; i < maxIter;i++) {
        eta = a * pow(b + pass*1.0*maxIter + i + 1, -gamma);
        
        for (k = 0; k < batchSize; k++) {
            idx = idxSample.gen();
            sampleBuffer[k] = idx;
            innerProdI = Xt.col(idx).dot(w);
            innerProdZ = Xt.col(idx).dot(wtilde);
            gradBuffer(k) = LogisticPartialGradient(innerProdI, 0) - LogisticPartialGradient(innerProdZ,0);
        }

        w = -eta *G + (1 - eta * lambda)*w;
        
        w = NOISY? w.array() + noise.gen():w;

        for (k = 0; k < batchSize; k++) {
            idx = sampleBuffer[k];
            w += (-eta * gradBuffer(k) / batchSize) * Xt.col(idx);
        }
        for (k = 0; k < batchSize; k++) {
            idx = sampleBuffer[k];
            w += -eta*gradBuffer[k]/batchSize*Xt.col(idx);
        }
        //compute error
        if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) {
            auto endTime = Clock::now();
            double telapsed = chrono::duration_cast<chrono::nanoseconds>(endTime-startTime).count()/BILLION;
            LogisticError(w, XtTest, yTest, pass + (i + 1)*1.0 / maxIter, telapsed, fp);
            epochCounter = (epochCounter + 1) % PRINT_FREQ;
            if (telapsed >= maxRunTime) {
                return 1;
            }
        }
    }
    delete[] sampleBuffer;
    return 0;
}