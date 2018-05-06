#ifndef COVTYPE_H
#define COVTYPE_H
#include <fstream>
#include <sstream>
#include "comm.h"
#include "SparseMat.h"
using namespace std;

void covtype_read(MatrixXd &Xt, VectorXd &y, MatrixXd &XtTest, VectorXd &yTest);
void covtype_binary_read(string full_path, SparseMatrix<double> &Xt, VectorXd &y, int setSize/*, SparseMatrix<double> &XtTest, VectorXd &yTest*/);
#endif