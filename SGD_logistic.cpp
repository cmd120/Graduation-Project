#include "include/SGD.h"

/*
SGD_logistic(w,Xt,y,lambda,eta,d,g);
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
int epochCounter;
FILE *fp;
auto startTime = Clock::now();

void SGD_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, \
     VectorXd &yTest, VectorXd &d, VectorXd &g, string filename, double lambda, double eta, \
    int maxIter, int batchSize, int pass, double a, double b, int gamma,  int maxRunTime) {
    
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
        flag = batchSize>=2?SGD_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, d, g, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, batchSize, maxRunTime):\
                            SGD_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, d, g, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);
        if (flag) {
            break;
        }
    }
    fclose(fp);

    auto endTime = Clock::now();
    cout << "duration: " << chrono::duration_cast<chrono::nanoseconds>(endTime-startTime).count()/BILLION << endl;

    return;
}



int SGD_LogisticInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime)
{
    long i, idx, j;
    double innerProd = 0 , tmpDelta, eta;
    Noise idxSample(0,nSamples-1);
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    for (i = 0; i < maxIter; i++) {
        eta = a * pow(b + i + 1, -gamma);
        idx = idxSample.gen();
        innerProd += w(j)*Xt.col(idx)(j);
        tmpDelta = LogisticPartialGradient(innerProd, y(idx));
        w = -eta*tmpDelta*Xt.col(idx)+(1-eta*lambda)*w;
        w = NOISY?w.array()+noise.gen():w;
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

int SGD_LogisticInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize, int maxRunTime)
{
    long i, idx, j, k;
    double innerProd, eta;

    VectorXd gradBuffer(batchSize);
    int* sampleBuffer = new int[batchSize];
    Noise idxSample(0,nSamples-1);
    for (i = 0; i < maxIter;i++) {
        eta = a * pow(b + i + 1, -gamma);
        Noise noise(0.0, sqrt(eta * 2 / nSamples));    
        for (k = 0; k < batchSize; k++) {
            idx = idxSample.gen();
            sampleBuffer[k] = idx;
            innerProd = Xt.col(idx).dot(w);
            gradBuffer(k) = LogisticPartialGradient(innerProd, y(idx));
        }

        w = (1-eta*lambda) * w;
        
        w = NOISY?w.array()+noise.gen():w;

        for (k = 0; k < batchSize; k++) {
            idx = sampleBuffer[k];
            w += (-eta * gradBuffer(k)  / batchSize)*Xt.col(idx);
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