#ifndef LOGISTIC_ERROR_H
#define LOGISTIC_ERROR_H
#include "comm.h"

using namespace Eigen;

ERRORCODE LogisticError(const VectorXd &w, const MatrixXd &Xt,
                        const VectorXd &y, double epoch, double telapsed,
                        FILE *fp);

ERRORCODE LogisticError(const VectorXd &w, const SparseMatrix<double> &Xt,
                        const VectorXd &y, double epoch, double telapsed,
                        FILE *fp);

#endif