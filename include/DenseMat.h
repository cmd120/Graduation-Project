#ifndef DENSE_H
#define DENSE_H

#include "Eigen/Dense"
#include "LogisticError.h"
#include "comm.h"
///
/// Compute the partial gradient vector for logistic regression.
/// \return A partial gradient vector
///
Eigen::VectorXd LogisticPartialGradient(Eigen::VectorXd &innerProdI,
                                        Eigen::VectorXd &y);
///
/// Compute the partial gradient for logistic regression.
/// \return Partial gradient result
///
double LogisticPartialGradient(double innerProdI, double y);
///
/// Compute the partial gradient vector for ridge regression.
/// \return A partial gradient vector
///
Eigen::VectorXd RidgePartialGradient(Eigen::VectorXd &innerProd,
                                     Eigen::VectorXd &y);
///
/// Compute the partial gradient for ridge regression.
/// \return Partial gradient result
///
double RidgePartialGradient(double innerProd, double y);
///
/// Set necessary environment before executing algorithm part, such as output
/// file, step size, batch size.
///
void algorithmInit(Eigen::MatrixXd &Xt, Eigen::VectorXd &w,
                   Eigen::MatrixXd &XtTest, Eigen::VectorXd &yTest,
                   double &lambda, double &eta, double &a, double &b,
                   double &gamma, int &maxIter, int &batchSize, int &passes,
                   int &maxRunTime, std::string &filename, int &datasetNum);
#endif