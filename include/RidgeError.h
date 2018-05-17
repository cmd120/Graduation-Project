#ifndef RIDGE_ERROR_H
#define RIDGE_ERROR_H
#include "comm.h"

ERRORCODE RidgeError(const Eigen::VectorXd &w, const Eigen::MatrixXd &Xt,
                     const Eigen::VectorXd &y, double epoch, double telapsed,
                     FILE *fp);

ERRORCODE RidgeError(const Eigen::VectorXd &w,
                     const Eigen::SparseMatrix<double> &Xt,
                     const Eigen::VectorXd &y, double epoch, double telapsed,
                     FILE *fp);

#endif