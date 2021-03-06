#ifndef SGD_H
#define SGD_H
#include "LogisticError.h"
#include "RidgeError.h"
#include "comm.h"
int SGD_LogisticInnerLoopSingle(Eigen::VectorXd &w, const Eigen::MatrixXd &Xt,
                                Eigen::VectorXd &y,
                                const Eigen::MatrixXd &XtTest,
                                Eigen::VectorXd &yTest, Eigen::VectorXd &d,
                                Eigen::VectorXd &g, double lambda, long maxIter,
                                int nSamples, int nVars, int pass, double a,
                                double b, double gamma, int maxRunTime);
int SGD_LogisticInnerLoopBatch(
    Eigen::VectorXd &w, const Eigen::MatrixXd &Xt, Eigen::VectorXd &y,
    const Eigen::MatrixXd &XtTest, Eigen::VectorXd &yTest, Eigen::VectorXd &d,
    Eigen::VectorXd &g, double lambda, long maxIter, int nSamples, int nVars,
    int pass, double a, double b, double gamma, int maxRunTime, int batchSize);
int SGD_LogisticInnerLoopSingle(
    Eigen::VectorXd &w, Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &y,
    int *innerIndices, int *outerStarts, Eigen::SparseMatrix<double> &XtTest,
    Eigen::VectorXd &yTest, double lambda, Eigen::VectorXd &d,
    Eigen::VectorXd &g, long maxIter, int nSamples, int nVars, int pass,
    double a, double b, double gamma, int maxRunTime);
int SGD_LogisticInnerLoopBatch(
    Eigen::VectorXd &w, Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &y,
    int *innerIndices, int *outerStarts, Eigen::SparseMatrix<double> &XtTest,
    Eigen::VectorXd &yTest, double lambda, Eigen::VectorXd &d,
    Eigen::VectorXd &g, long maxIter, int nSamples, int nVars, int pass,
    double a, double b, double gamma, int maxRunTime, int batchSize);
#endif