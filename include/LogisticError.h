#ifndef LOGISTIC_ERROR_H
#define LOGISTIC_ERROR_H
#include "comm.h"
#include "DenseMat.h"
#include "SparseMat.h"

using namespace Eigen;

void LogisticError(const VectorXd &w, const MatrixXd &Xt, const VectorXd &y, double epoch, double telapsed, FILE *fp);

void LogisticError(const VectorXd &w, const SparseMatrix<double> &Xt, const VectorXd &y, double epoch, double telapsed, FILE *fp);

//Sparse Matrix
// void LogisticError(const VectorXd &w, const SparseMatrix<double> &Xt, VectorXd &y, double epoch, double telapsed, FILE *fp)
// {
//     int nSamples,nVars;
//     long i, j;
//     double tmp,sumError=0;
//     double innerProd;
    
//     VectorXd tmpRes;    

//     nSamples = Xt.rows();
//     nVars = Xt.cols();
//     const int *innerIndices = Xt.innerIndexPtr();
//     int *outerStarts = new int[nVars];
//     InitOuterStarts(Xt, outerStarts);

//     for(i=0;i<nSamples;++i){
//         innerProd = 0;
//         for(j = outerStarts[i];j<(long)outerStarts[i+1];++j){
//             // Xt[j] ？？？
//             innerProd += w(outerStarts[j])*Xt[j]
//         }
//         tmpRes[i] = innerProd;
//     }

//     for(i = 0; i < nSamples; i++)
//     {
//         tmp = 1.0/(1 + exp(-tmpRes[i]));
//         sumError += y[i] * log(tmp) + (1 - y[i]) * log(1 - tmp);
//     }

//     return;
// }

#endif