#ifndef RIDGE_ERROR_H
#define RIDGE_ERROR_H
#include "comm.h"

using namespace Eigen;

ERRORCODE RidgeError(const VectorXd &w, const MatrixXd &Xt, const VectorXd &y,
                     double epoch, double telapsed, FILE *fp);

ERRORCODE RidgeError(const VectorXd &w, const SparseMatrix<double> &Xt,
                     const VectorXd &y, double epoch, double telapsed,
                     FILE *fp);

#endif