#ifndef RIDGE_ERROR_H
#define RIDGE_ERROR_H
#include "comm.h"
#include "DenseMat.h"
#include "SparseMat.h"

using namespace Eigen;

void RidgeError(const VectorXd &w, const MatrixXd &Xt, const VectorXd &y, double epoch, double telapsed, FILE *fp);
#endif