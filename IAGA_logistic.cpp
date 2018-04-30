#include "include/IAGA.h"
/*
IAGA_logistic(w,Xt,y,lambda,eta,d,g);
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
% yTest
% maxRunTime
% filename - saving results
*/

void IAGA_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, \
     VectorXd &yTest, VectorXd &d, VectorXd &g, string filename, double lambda, double eta, \
    int maxIter, int batchSize, int pass, double a, double b, int gamma, int maxRunTime) {
    
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

    for (int i = 0; i < pass; i++) {
        flag = batchSize>=2?IAGA_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, d, g, lambda, 2*nSamples, nSamples, nVars, pass, a, b, gamma, batchSize, maxRunTime):\
                            IAGA_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, d, g, lambda, 2*nSamples, nSamples, nVars, pass, a, b, gamma, maxRunTime);
        if (flag) {
            break;
        }
    }
    fclose(fp);

    auto endTime = Clock::now();
    cout << "duration: " << chrono::duration_cast<chrono::nanoseconds>(endTime-startTime).count()/BILLION << endl;

    return;
}

// void IAGA_logistic(VectorXd &w, const SparseMatrix<double> &Xt, int* innerIndices, int* outerStarts, const VectorXd &y, double lambda, double eta, VectorXd d, VectorXd g, \
//     int maxIter, int batchSize, int pass, int a, int b, int gamma, const MatrixXd &XtTest, \
//     const VectorXd &yTest, int maxRunTime, string filename){

//     int nVars, nSamples, flag;
//     int epochCounter = 0;
//     nVars = Xt.rows();
//     nSamples = Xt.cols();
//     FILE *fp = fopen(filename.c_str(), "a");
//     if (fp == NULL) {
//         cout << "Cannot write results to file: " << filename << endl;
//     }

//     LogisticError(w, XtTest, yTest, 0, 0, fp);
//     epochCounter = (epochCounter + 1) % PRINT_FREQ;

//     //为什么ret会在循环内部不断更新
//     for (int i = 0; i < pass; i++) {
//         flag = batchSize?InnerLoopBatchSparse(w, Xt, innerIndices, outerStarts, y, lambda, d, g, maxIter, nSamples, nVars, s, a, b, gamma):\
//                             InnerLoopSingleSparse(w, Xt, innerIndices, outerStarts, y, lambda, d, g, maxIter, nSamples, nVars, s, a, b, gamma);
//         if (flag) {
//             break;
//         }
//     }
// }


int IAGA_LogisticInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, const VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime){
    long i, idx, j;
    double innerProd=0, tmpDelta, eta;
    for (i = 0; i < maxIter; i++) {
        eta = a * pow(b + i + 1, -gamma);
        Noise noise(0.0,sqrt(eta*2/nSamples));
        idx = i % nSamples;
        innerProd = Xt.col(idx).dot(w);
        tmpDelta = LogisticPartialGradient(innerProd, y(idx));
        
        w = -eta/nSamples*d + (1-eta*lambda)*w;

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
int IAGA_LogisticInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, const VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d,  VectorXd &g, double lambda, long maxIter, int nSamples, int nVars,int pass, double a, double b, double gamma, int batchSize, int maxRunTime){
    long i, idx, j, k;
    double innerProd, eta;

    VectorXd gradBuffer(batchSize);
    int* sampleBuffer = new int[batchSize];
    
    Noise noise(0.0, sqrt(eta * 2 / nSamples));
    for (i = 0; i < maxIter;i++) {
        eta = a * pow(b + i + 1, -gamma);
        for (k = 0; k < batchSize; k++) {
            idx = (i*batchSize + k) % nSamples;
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

