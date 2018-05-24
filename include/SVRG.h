#ifndef SVRG_H
#define SVRG_H
#include "LogisticError.h"
#include "RidgeError.h"
#include "comm.h"
/// 
/// SVRG method(deal with dense dataset): batch size = 1
///
int SVRG_LogisticInnerLoopSingle(Eigen::VectorXd &w, const Eigen::MatrixXd &Xt,
                                 Eigen::VectorXd &y,
                                 const Eigen::MatrixXd &XtTest,
                                 Eigen::VectorXd &yTest,
                                 Eigen::VectorXd &wtilde, Eigen::VectorXd &G,
                                 double lambda, long maxIter, int nSamples,
                                 int nVars, int pass, double a, double b,
                                 double gamma, int maxRunTime);
/// 
/// SVRG method(deal with dense dataset): batch size > 1
///
int SVRG_LogisticInnerLoopBatch(Eigen::VectorXd &w, const Eigen::MatrixXd &Xt,
                                Eigen::VectorXd &y,
                                const Eigen::MatrixXd &XtTest,
                                Eigen::VectorXd &yTest, Eigen::VectorXd &wtilde,
                                Eigen::VectorXd &G, double lambda, long maxIter,
                                int nSamples, int nVars, int pass, double a,
                                double b, double gamma, int maxRunTime,
                                int batchSize);
/// 
/// SVRG method(deal with sparse dataset): batch size = 1
///
int SVRG_LogisticInnerLoopSingle(
    Eigen::VectorXd &w, Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &y,
    int *innerIndices, int *outerStarts, Eigen::SparseMatrix<double> &XtTest,
    Eigen::VectorXd &yTest, Eigen::VectorXd &wtilde, Eigen::VectorXd &G,
    double lambda, long maxIter, int nSamples, int nVars, int pass, double a,
    double b, double gamma, int maxRunTime);
/// 
/// SVRG method(deal with sparse dataset): batch size > 1
///
int SVRG_LogisticInnerLoopBatch(
    Eigen::VectorXd &w, Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &y,
    int *innerIndices, int *outerStarts, Eigen::SparseMatrix<double> &XtTest,
    Eigen::VectorXd &yTest, Eigen::VectorXd &wtilde, Eigen::VectorXd &G,
    double lambda, long maxIter, int nSamples, int nVars, int pass, double a,
    double b, double gamma, int maxRunTime, int batchSize);
#endif