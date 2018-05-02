#include "include/SIG.h"

/*
SIG_logistic(w,Xt,y,lambda,eta,wtilde,G);
% w(p,1) - updated in place
% Xt(p,n) - real, can be sparse
% y(n,1) - {0,1}
% lambda - scalar regularization param
% wtilde(p,1) - initial approximation of average gradient (should be sum of previous gradients)
% G(n,1) - previous derivatives of loss
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
void SIG_init(MatrixXd &Xt, VectorXd &w, MatrixXd &XtTest, VectorXd &yTest, double &lambda, double &eta, double &a, double &b, double &gamma,\
    int &maxIter, int &batchSize, int &passes, int &maxRunTime, string &filename){
    startTime = Clock::now();
    cout << "Input batchSize: " << endl;
    cin >> batchSize;
    filename = "SIG_output_"+to_string(batchSize);
    fp = fopen(filename.c_str(), "a");
    if (fp == NULL) {
        cout << "Cannot write results to file: " << filename << endl;
    }
    LogisticError(w, XtTest, yTest, 0, 0, fp);
    epochCounter = (epochCounter + 1) % PRINT_FREQ;
    lambda = 1/Xt.cols();
    eta = 0.1;
    a = batchSize>=2?4:1e-1;
    b = 0;
    gamma = 0;
    maxIter = 2*Xt.cols();
    passes = 10;
    maxRunTime = 60;
    return;
}
// void SIG_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd y, const MatrixXd &XtTest, \
//      VectorXd &yTest, VectorXd& wtilde, VectorXd& G, string filename, double lambda, double eta, \
//     int maxIter, int batchSize, int pass, double a, double b, double gamma,  int maxRunTime) {
    
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
//         flag = batchSize>=2?SIG_LogisticInnerLoopBatchDense(w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, batchSize, maxRunTime):\
//                             SIG_LogisticInnerLoopSingleDense(w, Xt, y, XtTest, yTest, wtilde, G, lambda, maxIter, nSamples, nVars, pass, a, b, gamma, maxRunTime);
//         if (flag) {
//             break;
//         }
//     }
//     fclose(fp);

//     auto endTime = Clock::now();
//     cout << "duration: " << chrono::duration_cast<chrono::nanoseconds>(endTime-startTime).count()/BILLION << endl;

//     return;
// }



int SIG_LogisticInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &wtilde, VectorXd &G, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime)
{
    long i, idx, j;
    double innerProdI = 0 ,innerProdZ = 0, tmpDelta, eta, telapsed;
    auto endTime = Clock::now();
    for (i = 0; i < maxIter; i++) {
        eta = a * pow(b + pass*1.0*maxIter + i + 1, -gamma);
        Noise noise(0.0, sqrt(eta * 2 / nSamples));
        idx = i % nSamples;
        for(j=0;j<nVars;++j){
            innerProdI += w(j)*Xt.col(idx)(j);
            innerProdZ += wtilde(j) * Xt.col(idx)(j);
        }
        tmpDelta = LogisticPartialGradient(innerProdI, 0) - LogisticPartialGradient(innerProdZ, 0);
        w = -eta*G+(1-eta*lambda)*w;
        w = NOISY?w.array()+noise.gen():w;
        w = w + (-eta) * tmpDelta * Xt.col(idx);
        //compute error
        if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) {
            endTime = Clock::now();
            telapsed = chrono::duration_cast<chrono::nanoseconds>(endTime-startTime).count()/BILLION;
            LogisticError(w, XtTest, yTest, pass + (i + 1)*1.0 / maxIter, telapsed, fp);
            epochCounter = (epochCounter + 1) % PRINT_FREQ;
            if (telapsed >= maxRunTime) {
                return 1;
            }
        }
    }
    return 0;
}

int SIG_LogisticInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &wtilde, VectorXd &G, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime, int batchSize)
{
    long i, idx, j, k;
    double innerProdI=0,innerProdZ=0, eta, telapsed;
    auto endTime = Clock::now();
    VectorXd gradBuffer(batchSize);
    int* sampleBuffer = new int[batchSize];
    
    for (i = 0; i < maxIter;i++) {
        eta = a * pow(b + pass*1.0*maxIter + i + 1, -gamma);
        Noise noise(0.0, sqrt(eta * 2 / nSamples));
        for (k = 0; k < batchSize; k++) {
            idx = (i * batchSize + k ) % nSamples;
            sampleBuffer[k] = idx;
            innerProdI = Xt.col(idx).dot(w);
            innerProdZ = Xt.col(idx).dot(wtilde);
            gradBuffer(k) = LogisticPartialGradient(innerProdI, 0) - LogisticPartialGradient(innerProdZ, 0);
        }

        w = -eta * G + (1 - eta * lambda)*w;
        
        w = NOISY?w.array()+noise.gen():w;

        for (k = 0; k < batchSize; k++) {
            idx = sampleBuffer[k];
            w += (-eta * gradBuffer(k) / batchSize)*Xt.col(idx);
        }
        //compute error
        if ((i + 1) % maxIter == maxIter * epochCounter / PRINT_FREQ) {
            endTime = Clock::now();
            telapsed = chrono::duration_cast<chrono::nanoseconds>(endTime-startTime).count()/BILLION;
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