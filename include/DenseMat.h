#ifndef DENSE_H
#define DENSE_H

#include "comm.h"
#include "LogisticError.h"
using namespace std;
using namespace Eigen;

VectorXd LogisticPartialGradient(VectorXd &innerProdI, VectorXd &y);
double LogisticPartialGradient(double innerProdI, double y);
VectorXd RidgePartialGradient(VectorXd &innerProd, VectorXd &y);
double RidgePartialGradient(double innerProd, double y);
void algorithmInit(MatrixXd &Xt, VectorXd &w, MatrixXd &XtTest, VectorXd &yTest,
                   double &lambda, double &eta, double &a, double &b,
                   double &gamma, int &maxIter, int &batchSize, int &passes,
                   int &maxRunTime, string &filename, int &datasetNum);
#endif