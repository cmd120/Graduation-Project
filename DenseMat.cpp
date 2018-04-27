#include "include/DenseMat.h"

VectorXd LogisticPartialGradient(VectorXd &innerProd, VectorXd &y)
{
	return (1+(-innerProd).array().exp()).inverse() - y.array();
    
}
double LogisticPartialGradient(double innerProd, double y) {
	return 1 / (1 + exp(-innerProd)) - y;
}

VectorXd RidgePartialGradient(VectorXd &innerProd, VectorXd &y)
{
	return innerProd - y;
    
}
double RidgePartialGradient(double innerProd, double y) {
	return innerProd - y;
}
