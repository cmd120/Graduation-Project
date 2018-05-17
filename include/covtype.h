#ifndef COVTYPE_H
#define COVTYPE_H
#include <fstream>
#include <sstream>
#include "SparseMat.h"
#include "comm.h"

const int trainSetSize = 100;
const int testSetSize = 20;
void covtype_read(Eigen::MatrixXd &Xt, Eigen::VectorXd &y,
                  Eigen::MatrixXd &XtTest, Eigen::VectorXd &yTest);
// void covtype_binary_read(Eigen::SparseMatrix<double> &Xt, Eigen::VectorXd &y,
// int setSize, string full_path="covtype.libsvm.binary"/*,
// Eigen::SparseMatrix<double> &XtTest, Eigen::VectorXd &yTest*/);
#endif