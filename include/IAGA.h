#ifndef IAGA_H
#define IAGA_H
#include "LogisticError.h"
#include "RidgeError.h"
#include "comm.h"
int IAGA_LogisticInnerLoopSingle(
    Eigen::VectorXd &w, const Eigen::MatrixXd &Xt, const Eigen::VectorXd &y,
    const Eigen::MatrixXd &XtTest, Eigen::VectorXd &yTest, Eigen::VectorXd &d,
    Eigen::VectorXd &g, double lambda, long maxIter, int nSamples, int nVars,
    int pass, double a, double b, double gamma, int maxRunTime);
int IAGA_LogisticInnerLoopBatch(
    Eigen::VectorXd &w, const Eigen::MatrixXd &Xt, const Eigen::VectorXd &y,
    const Eigen::MatrixXd &XtTest, Eigen::VectorXd &yTest, Eigen::VectorXd &d,
    Eigen::VectorXd &g, double lambda, long maxIter, int nSamples, int nVars,
    int pass, double a, double b, double gamma, int maxRunTime, int batchSize);
int IAGA_LogisticInnerLoopSingle(
    Eigen::VectorXd &w, Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &y,
    int *innerIndices, int *outerStarts, Eigen::SparseMatrix<double> &XtTest,
    Eigen::VectorXd &yTest, double lambda, Eigen::VectorXd &d,
    Eigen::VectorXd &g, long maxIter, int nSamples, int nVars, int pass,
    double a, double b, double gamma, int maxRunTime);
int IAGA_LogisticInnerLoopBatch(
    Eigen::VectorXd &w, Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &y,
    int *innerIndices, int *outerStarts, Eigen::SparseMatrix<double> &XtTest,
    Eigen::VectorXd &yTest, double lambda, Eigen::VectorXd &d,
    Eigen::VectorXd &g, long maxIter, int nSamples, int nVars, int pass,
    double a, double b, double gamma, int maxRunTime, int batchSize);
#endif