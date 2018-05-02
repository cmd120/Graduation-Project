#ifndef SPARSE_H
#define SPARSE_H

#include "comm.h"

using namespace std;
using namespace Eigen;

int IsSparse(vector<double> &mat);
void InitOuterStarts(const SparseMatrix<double> &mat, int *outerStarts);

#endif