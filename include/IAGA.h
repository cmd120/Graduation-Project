#ifndef IAGA_H
#define IAGA_H
#include "LogisticError.h"
#include "RidgeError.h"
#include "comm.h"
int IAGA_LogisticInnerLoopSingle(VectorXd &w, const MatrixXd &Xt,
                                 const VectorXd &y, const MatrixXd &XtTest,
                                 VectorXd &yTest, VectorXd &d, VectorXd &g,
                                 double lambda, long maxIter, int nSamples,
                                 int nVars, int pass, double a, double b,
                                 double gamma, int maxRunTime);
int IAGA_LogisticInnerLoopBatch(VectorXd &w, const MatrixXd &Xt,
                                const VectorXd &y, const MatrixXd &XtTest,
                                VectorXd &yTest, VectorXd &d, VectorXd &g,
                                double lambda, long maxIter, int nSamples,
                                int nVars, int pass, double a, double b,
                                double gamma, int maxRunTime, int batchSize);
int IAGA_LogisticInnerLoopSingle(VectorXd &w, SparseMatrix<double> &Xt,
                                 VectorXd &y, int *innerIndices,
                                 int *outerStarts, SparseMatrix<double> &XtTest,
                                 VectorXd &yTest, double lambda, VectorXd &d,
                                 VectorXd &g, long maxIter, int nSamples,
                                 int nVars, int pass, double a, double b,
                                 double gamma, int maxRunTime);
int IAGA_LogisticInnerLoopBatch(VectorXd &w, SparseMatrix<double> &Xt,
                                VectorXd &y, int *innerIndices,
                                int *outerStarts, SparseMatrix<double> &XtTest,
                                VectorXd &yTest, double lambda, VectorXd &d,
                                VectorXd &g, long maxIter, int nSamples,
                                int nVars, int pass, double a, double b,
                                double gamma, int maxRunTime, int batchSize);
#endif