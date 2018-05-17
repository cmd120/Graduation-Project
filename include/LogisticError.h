#ifndef LOGISTIC_ERROR_H
#define LOGISTIC_ERROR_H
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "comm.h"

ERRORCODE LogisticError(const Eigen::VectorXd &w, const Eigen::MatrixXd &Xt,
                        const Eigen::VectorXd &y, double epoch, double telapsed,
                        FILE *fp);

ERRORCODE LogisticError(const Eigen::VectorXd &w,
                        const Eigen::SparseMatrix<double> &Xt,
                        const Eigen::VectorXd &y, double epoch, double telapsed,
                        FILE *fp);

#endif