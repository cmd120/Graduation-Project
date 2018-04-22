#ifndef DENSE_H
#define DENSE_H

#include "comm.h"

using namespace Eigen;

VectorXd LogisticPartialGradient(VectorXd &innerProdI, VectorXd &y);
double LogisticPartialGradient(double innerProdI, double y);
VectorXd RidgePartialGradient(VectorXd &innerProd, VectorXd &y);
double RidgePartialGradient(double innerProd, double y);

#endif