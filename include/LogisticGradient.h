#ifndef LOGISTIC_GRADIENT_H
#define LOGISTIC_GRADIENT_H
#include "comm.h"
using namespace Eigen;
using namespace std;
void LogisticGradient(VectorXd &wtilde, VectorXd &G, const MatrixXd &Xt,
                      VectorXd &y);
void LogisticGradient(VectorXd &wtilde, VectorXd &G, int *innerIndices,
                      int *outerStarts, const SparseMatrix<double> &Xt,
                      VectorXd &y);
#endif