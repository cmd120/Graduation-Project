#include "include/DenseMat.h"

VectorXd LogisticPartialGradient(VectorXd &innerProdI, VectorXd &y)
{
	return 1/(1+(-innerProdI).array().exp()) - y.array();
    
    // return 1/(1 + exp(-innerProdI)) - y;
}
double LogisticPartialGradient(double innerProdI, double y) {
	return 1 / (1 + exp(-innerProdI)) - y;
}

VectorXd RidgePartialGradient(VectorXd &innerProd, VectorXd &y)
{
	return innerProd - y;
    
    // return 1/(1 + exp(-innerProdI)) - y;
}
double RidgePartialGradient(double innerProd, double y) {
	return innerProd - y;
}
