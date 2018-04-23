#include "include/SparseMat.h"

int SPARSE = 0;
// template <typename T>
// void buildProblem(vector<T> &coefficients, int rows, int cols, )

// // if not specified, we make sparse matrix compressed
// mat.makeCompressed();

//create outerStarts array(), jc is counterpart in matlab
void InitOuterStarts(const SparseMatrix<double> &mat, int* outerStarts){
	*outerStarts = 0;
	int count=0;
	for(int k=0;k<mat.outerSize();++k){
		SparseMatrix<double>::InnerIterator it(mat,k);
		int nextColFirstIndex=0;
		if(it){
			while(it){
				nextColFirstIndex++;
				++it;
			}
			outerStarts[count+1] = outerStarts[count] + nextColFirstIndex; 
			count++;
		}
	}
	for(int i=0;i<count+1;++i){
		cout << outerStarts[i] <<endl;
	}
}

// //ir is counterpart of mat.innerIndexPtr() in matlab
// for(int i=0;i<mat.nonZeros();++i){
// 	cout << mat.innerIndexPtr()[i] << endl;
// }
// int main(){
// 	return 0;
// }