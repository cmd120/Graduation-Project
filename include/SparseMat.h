#ifndef SPARSE_H
#define SPARSE_H

#include "Eigen/Sparse"
#include "LogisticError.h"
#include "comm.h"

int issparse(std::vector<double> &mat);
void InitOuterStarts(const Eigen::SparseMatrix<double> &mat, int *outerStarts);
void algorithmInit(Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &w,
                   Eigen::SparseMatrix<double> &XtTest, Eigen::VectorXd &yTest,
                   double &lambda, double &eta, double &a, double &b,
                   double &gamma, int &maxIter, int &batchSize, int &passes,
                   int &maxRunTime, std::string &filename, int &datasetNum);
#endif