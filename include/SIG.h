#ifndef SIG_H
#define SIG_H
#include "LogisticError.h"
#include "RidgeError.h"
#include "comm.h"
int SIG_LogisticInnerLoopSingle(Eigen::VectorXd &w, const Eigen::MatrixXd &Xt,
                                Eigen::VectorXd &y,
                                const Eigen::MatrixXd &XtTest,
                                Eigen::VectorXd &yTest, Eigen::VectorXd &wtilde,
                                Eigen::VectorXd &G, double lambda, long maxIter,
                                int nSamples, int nVars, int pass, double a,
                                double b, double gamma, int maxRunTime);
int SIG_LogisticInnerLoopBatch(Eigen::VectorXd &w, const Eigen::MatrixXd &Xt,
                               Eigen::VectorXd &y,
                               const Eigen::MatrixXd &XtTest,
                               Eigen::VectorXd &yTest, Eigen::VectorXd &wtilde,
                               Eigen::VectorXd &G, double lambda, long maxIter,
                               int nSamples, int nVars, int pass, double a,
                               double b, double gamma, int maxRunTime,
                               int batchSize);
int SIG_LogisticInnerLoopSingle(
    Eigen::VectorXd &w, Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &y,
    int *innerIndices, int *outerStarts, Eigen::SparseMatrix<double> &XtTest,
    Eigen::VectorXd &yTest, Eigen::VectorXd &wtilde, Eigen::VectorXd &G,
    double lambda, long maxIter, int nSamples, int nVars, int pass, double a,
    double b, double gamma, int maxRunTime);
int SIG_LogisticInnerLoopBatch(
    Eigen::VectorXd &w, Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &y,
    int *innerIndices, int *outerStarts, Eigen::SparseMatrix<double> &XtTest,
    Eigen::VectorXd &yTest, Eigen::VectorXd &wtilde, Eigen::VectorXd &G,
    double lambda, long maxIter, int nSamples, int nVars, int pass, double a,
    double b, double gamma, int maxRunTime, int batchSize);
#endif