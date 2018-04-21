#ifndef DENSE_H
#define DENSE_H

#include "comm.h"

using namespace Eigen;

VectorXd LogisticPartialGradient(VectorXd &innerProdI, VectorXd &y)
{
	return 1/(1+(-innerProdI).array().exp()) - y.array();
    
    // return 1/(1 + exp(-innerProdI)) - y;
}
double LogisticPartialGradient(double innerProdI, double y) {
	return 1 / (1 + exp(-innerProdI)) - y;
}

#endif