#ifndef SPARSE_H
#define SPARSE_H

#include "comm.h"

using namespace std;
using namespace Eigen;

template <typename T>
int IsSparseMat(vector<T> &mat){
	int ret;
	long count = 0;
	for(long i=0;i < mat.size(); ++i){
		count = (mat[i] - 0)  < ACCURACY ? count : count + 1;
	}
	ret = (double)count / mat.size() > 0.05 ? 0 : 1;
	return ret;
}

void InitOuterStarts(const SparseMatrix<double> &mat, int *outerStarts);

#endif