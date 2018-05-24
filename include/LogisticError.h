#ifndef LOGISTIC_ERROR_H
#define LOGISTIC_ERROR_H
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "comm.h"
///
/// Compute the logsistic regression error for dense dataset.
///
ERRORCODE LogisticError(const Eigen::VectorXd &w, const Eigen::MatrixXd &Xt,
                        const Eigen::VectorXd &y, double epoch, double telapsed,
                        FILE *fp);
///
/// Compute the logistic regression error for sparse datset.
///
ERRORCODE LogisticError(const Eigen::VectorXd &w,
                        const Eigen::SparseMatrix<double> &Xt,
                        const Eigen::VectorXd &y, double epoch, double telapsed,
                        FILE *fp);

#endif