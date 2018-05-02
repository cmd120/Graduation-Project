#ifndef COVTYPE_H
#define COVTYPE_H
#include <fstream>
#include <sstream>
#include "comm.h"
#include "SparseMat.h"
using namespace std;

void covtype_read(MatrixXd &Xt, VectorXd &y, MatrixXd &XtTest, VectorXd &yTest);

#endif