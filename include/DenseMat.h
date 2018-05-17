#ifndef DENSE_H
#define DENSE_H

#include "Eigen/Dense"
#include "LogisticError.h"
#include "comm.h"

Eigen::VectorXd LogisticPartialGradient(Eigen::VectorXd &innerProdI,
                                        Eigen::VectorXd &y);
double LogisticPartialGradient(double innerProdI, double y);
Eigen::VectorXd RidgePartialGradient(Eigen::VectorXd &innerProd,
                                     Eigen::VectorXd &y);
double RidgePartialGradient(double innerProd, double y);
void algorithmInit(Eigen::MatrixXd &Xt, Eigen::VectorXd &w,
                   Eigen::MatrixXd &XtTest, Eigen::VectorXd &yTest,
                   double &lambda, double &eta, double &a, double &b,
                   double &gamma, int &maxIter, int &batchSize, int &passes,
                   int &maxRunTime, std::string &filename, int &datasetNum);
#endif