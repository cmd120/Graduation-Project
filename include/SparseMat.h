#ifndef SPARSE_H
#define SPARSE_H

#include "comm.h"
#include "LogisticError.h"
using namespace std;
using namespace Eigen;

int issparse(vector<double> &mat);
void InitOuterStarts(const SparseMatrix<double> &mat, int *outerStarts);
void algorithmInit(SparseMatrix<double> &Xt, VectorXd &w,
                   SparseMatrix<double> &XtTest, VectorXd &yTest,
                   double &lambda, double &eta, double &a, double &b,
                   double &gamma, int &maxIter, int &batchSize, int &passes,
                   int &maxRunTime, string &filename, int &datasetNum);
#endif