#include "include/SAGA.h"

/*
SAGA_logistic(w,Xt,y,lambda,eta,d,g);
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

// void SAGA_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd y, const MatrixXd &XtTest, \
//      VectorXd &yTest, VectorXd d, VectorXd g, string filename, double lambda, double eta, \
//     int maxIter, int batchSize, int pass, int a, int b, int gamma,  int maxRunTime) {
    
//     startTime = Clock::now();
    
//     int nVars, nSamples, flag;
//     int epochCounter = 0;
//     nVars = Xt.rows();
//     nSamples = Xt.cols();
//     FILE *fp = fopen(filename.c_str(), "a");
//     if (fp == NULL) {
//         cout << "Cannot write results to file: " << filename << endl;
//     }
//     epochCounter = 0;
//     LogisticError(w, XtTest, yTest, 0, 0, fp);
//     epochCounter = (epochCounter + 1) % PRINT_FREQ;
//     //为什么ret会在循环内部不断更新
//     for (int i = 0; i < pass; i++) {
//         flag = batchSize?SAGA_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, d, g, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, batchSize, maxRunTime):\
//                             SAGA_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, d, g, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);
//         if (flag) {
//             break;
//         }
//     }
//     fclose(fp);

//     auto endTime = Clock::now();
//     cout << "duration: " << chrono::duration_cast<chrono::nanoseconds>(endTime-startTime).count()/BILLION << endl;

//     return;
// }



int SAGA_LogisticInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd& y, const MatrixXd & XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime)
{
    long i, idx, j;
    double innerProd = 0 , tmpDelta, eta;
    Noise idxSample(0,nSamples-1);
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    for (i = 0; i < maxIter; i++) {
        eta = a * pow(b + i + 1, -gamma);
        idx = idxSample.gen();
        
        innerProd = Xt.col(idx).dot(w);
        tmpDelta = LogisticPartialGradient(innerProd, y(idx));
        w = -eta/nSamples*d+(1-eta*lambda)*w;
        w = NOISY?w.array()+noise.gen():w;
        w = w + (-eta) * (tmpDelta - g(idx)) * Xt.col(idx);
        d = d + (tmpDelta - g(idx)) * Xt.col(idx);
        g(idx) = tmpDelta;

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

int SAGA_LogisticInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize, int maxRunTime)
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
            gradBuffer(k) = LogisticPartialGradient(innerProd, y(idx));
        }

        w = -eta / nSamples * d + (1 - eta * lambda)*w;
        
        w = NOISY?w.array()+noise.gen():w;

        for (k = 0; k < batchSize; k++) {
            idx = sampleBuffer[k];
            w += (-eta * (gradBuffer(k) - g(idx)) / batchSize)*Xt.col(idx);
        }
        for (k = 0; k < batchSize; k++) {
            idx = sampleBuffer[k];
            d += (gradBuffer(k) - g(idx))*Xt.col(idx);
            g(idx) = gradBuffer(k);
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