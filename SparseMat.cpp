#include "include/SparseMat.h"

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
	// DEBUG
	// for(int i=0;i<count+1;++i){
	// 	cout << outerStarts[i] <<endl;
	// }
	return;
}

int issparse(vector<double> &mat){
	int ret;
	long count = 0;
	// cout << "mat size: " << mat.size() << endl;
	for(long i=0;i < mat.size(); ++i){
		count = (mat[i] - 0)  < ACCURACY ? count : count + 1;
	}
	ret = count >= mat.size()/2 ? 0 : 1;
	return ret;
}

// //ir is counterpart of mat.innerIndexPtr() in matlab
// for(int i=0;i<mat.nonZeros();++i){
// 	cout << mat.innerIndexPtr()[i] << endl;
// }
// int main(){
// 	return 0;
// }


// int outerIndexPtr[cols+1];
// int innerIndices[nnz];
// double values[nnz];
// Map<SparseMatrix<double> > sm1(rows,cols,nnz,outerIndexPtr, // read-write
//                                innerIndices,values);
// Map<const SparseMatrix<double> > sm2(...);                  // read-only
// int main(){
// 	SparseMatrix<double> mat(7,3);
// 	vector<Triplet<double>> tripletList;
// 	tripletList.push_back(Triplet<double>(1,0,1));
// 	tripletList.push_back(Triplet<double>(4,0,1));
// 	tripletList.push_back(Triplet<double>(2,1,1));
// 	tripletList.push_back(Triplet<double>(1,2,2));
// 	tripletList.push_back(Triplet<double>(4,2,1));
// 	tripletList.push_back(Triplet<double>(5,2,1));
// 	mat.setFromTriplets(tripletList.begin(),tripletList.end());
// 	int *innerIndices = mat.innerIndexPtr();
// 	for(int i=0;i<6;++i){
// 		cout << innerIndices[i] << endl;
// 	}
// 	int *outerStarts = new int[mat.cols()];
// 	InitOuterStarts(mat,outerStarts);
// 	cout << "pass" << endl;
// 	return 0;
// }