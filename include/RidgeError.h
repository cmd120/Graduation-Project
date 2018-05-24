#ifndef RIDGE_ERROR_H
#define RIDGE_ERROR_H
#include "comm.h"
///
/// Compute the ridge regression error for dense dataset.
///
ERRORCODE RidgeError(const Eigen::VectorXd &w, const Eigen::MatrixXd &Xt,
                     const Eigen::VectorXd &y, double epoch, double telapsed,
                     FILE *fp);
///
/// Compute the ridge regression error for sparse dataset.
///
ERRORCODE RidgeError(const Eigen::VectorXd &w,
                     const Eigen::SparseMatrix<double> &Xt,
                     const Eigen::VectorXd &y, double epoch, double telapsed,
                     FILE *fp);

#endif