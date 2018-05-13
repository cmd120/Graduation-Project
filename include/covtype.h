#ifndef COVTYPE_H
#define COVTYPE_H
#include <fstream>
#include <sstream>
#include "comm.h"
#include "SparseMat.h"
using namespace std;

const int trainSetSize = 100;
const int testSetSize = 20;
void covtype_read(MatrixXd &Xt, VectorXd &y, MatrixXd &XtTest, VectorXd &yTest);
// void covtype_binary_read(SparseMatrix<double> &Xt, VectorXd &y, int setSize, string full_path="covtype.libsvm.binary"/*, SparseMatrix<double> &XtTest, VectorXd &yTest*/);
#endif