#ifndef LOGISTIC_GRADIENT_H
#define LOGISTIC_GRADIENT_H
#include "comm.h"
void LogisticGradient(Eigen::VectorXd &wtilde, Eigen::VectorXd &G,
                      const Eigen::MatrixXd &Xt, Eigen::VectorXd &y);
void LogisticGradient(Eigen::VectorXd &wtilde, Eigen::VectorXd &G,
                      int *innerIndices, int *outerStarts,
                      const Eigen::SparseMatrix<double> &Xt,
                      Eigen::VectorXd &y);
#endif