#ifndef SAG_H
#define SAG_H
#include "comm.h"
#include "LogisticError.h"
#include "RidgeError.h"
using namespace Eigen;
int SAG_LogisticInnerLoopSingle(VectorXd &w, const MatrixXd &Xt, VectorXd y,
                                const MatrixXd &XtTest, VectorXd yTest,
                                VectorXd &d, VectorXd &g, double lambda,
                                long maxIter, int nSamples, int nVars, int pass,
                                double a, double b, double gamma,
                                int maxRunTime);
int SAG_LogisticInnerLoopBatch(VectorXd &w, const MatrixXd &Xt, VectorXd y,
                               const MatrixXd &XtTest, VectorXd yTest,
                               VectorXd &d, VectorXd &g, double lambda,
                               long maxIter, int nSamples, int nVars, int pass,
                               double a, double b, double gamma, int maxRunTime,
                               int batchSize);
int SAG_LogisticInnerLoopSingle(VectorXd &w, SparseMatrix<double> &Xt,
                                VectorXd &y, int *innerIndices,
                                int *outerStarts, SparseMatrix<double> &XtTest,
                                VectorXd &yTest, double lambda, VectorXd &d,
                                VectorXd &g, long maxIter, int nSamples,
                                int nVars, int pass, double a, double b,
                                double gamma, int maxRunTime);
int SAG_LogisticInnerLoopBatch(VectorXd &w, SparseMatrix<double> &Xt,
                               VectorXd &y, int *innerIndices, int *outerStarts,
                               SparseMatrix<double> &XtTest, VectorXd &yTest,
                               double lambda, VectorXd &d, VectorXd &g,
                               long maxIter, int nSamples, int nVars, int pass,
                               double a, double b, double gamma, int maxRunTime,
                               int batchSize);
#endif 