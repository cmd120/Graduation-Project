#include "include/comm.h"
#include "include/DenseMat.h"
#include "include/SparseMat.h"
#include "include/MNIST_Read.h"
#include "include/IAG.h"


int main()
{	
    // string test_image_path = "test-images";
    // string test_label_path = "test-labels";
    // vector<BYTE> test_image_dataset = read_mnist_images(test_image_path);
    // vector<BYTE> test_label_dataset = read_mnist_labels(test_label_path);
    // // cout << "SPARSE: " << IsSparseMat<BYTE>(test_image_dataset) << endl;
    // // cout << "SPARSE: " << endl;
    // // MatrixXd Mdata(_dataset.data(),28,_dataset.size()/28);
    // vector<double> Xt_data(test_image_dataset.begin(),test_image_dataset.end());
    // vector<double> y_data(test_label_dataset.begin(),test_label_dataset.end());
    // Map<Matrix<double,Dynamic,Dynamic,RowMajor>> Mtdata(Xt_data.data(), Xt_data.size()/28, 28);
    // Map<Matrix<double,Dynamic,Dynamic,ColMajor>> Vtdata(y_data.data(), y_data.size(), 1);
    // VectorXd w,d,g;
    // w = MatrixXd::Zero(Mtdata.rows(),1);
    // w = VectorXd(w);
    // d = MatrixXd::Zero(Mtdata.rows(),1);
    // d = VectorXd(d);
    // g = MatrixXd::Zero(Mtdata.rows(),1);
    // g = VectorXd(g);
    // string filename = "dong";
    // MatrixXd Xt_test(Mtdata);
    // VectorXd y_test(Vtdata);
    // cout << (int)Vtdata(0) <<endl;
    // IAG_logistic(w, Mtdata, Vtdata, Xt_test, y_test, d, g, filename);
    VectorXd a(3);
    a(0) = 1;
    a(1) = 2;
    a(2) = 3;
    return 0;
    // IAG_logistic(w, Xt, y, Xt_test, y_test, d, g, filename);
	// vector<unsigned char> aa;
	// aa.push_back(0);
	// vector<unsigned char> vector_a;
	// vector_a.push_back(1);
	// // cout << test<unsigned char>(vector_a) << endl;
	// cout << IsSparseMat(aa) << endl;
	// MatrixXd m(3,2);
	// m << 1,2,3,4,5,6;
	// cout  << "m: " << m << endl;
	// cout << "m(2): " << m.col(1) << endl;
	// VectorXd a(3);
	// a << 1,2,3;
	// // cout << a-a << endl;
	// cout <<m.col(1) << endl;
	// cout << a+m.col(1) << endl;
	// vector<Triplet<double>> tripletList;
	// tripletList.reserve(10);
	// tripletList.push_back(Triplet<double>(0,0,3));
	// tripletList.push_back(Triplet<double>(1,1,3.1));
	// tripletList.push_back(Triplet<double>(2,2,3.2));
	// tripletList.push_back(Triplet<double>(5,2,4.1));
	// tripletList.push_back(Triplet<double>(3,3,3.3));
	// tripletList.push_back(Triplet<double>(4,4,3.4));
	// SparseMatrix<double> mat(50,50);
	// mat.setFromTriplets(tripletList.begin(),tripletList.end());
	// cout << mat.rows() << endl;
	// cout << mat.cols() << endl;
	// cout << mat.nonZeros() << endl;
	// //how we create outerStarts array(), jd is counterpart in matlab
	// mat.makeCompressed();
	// int *outerStarts = new int[mat.cols()];//mat.cols() is the maximum
	// *outerStarts = 0;
	// int count=0;
	// for(int k=0;k<mat.outerSize();++k){
	// 	SparseMatrix<double>::InnerIterator it(mat,k);
	// 	int nextColFirstIndex=0;
	// 	if(it){
	// 		while(it){
	// 			nextColFirstIndex++;
	// 			++it;
	// 		}
	// 		outerStarts[count+1] = outerStarts[count] + nextColFirstIndex; 
	// 		count++;
	// 	}
	// }
	// // cout << count << endl;
	// for(int i=0;i<count+1;++i){
	// 	cout << outerStarts[i] <<endl;
	// }
	// delete[] outerStarts;
	// //ir is counterpart of mat.innerIndexPtr() in matlab
	// for(int i=0;i<mat.nonZeros();++i){
	// 	cout << mat.innerIndexPtr()[i] << endl;
	// }
	// cout << mat.OuterStarts[1] <<endl;
	// VectorXd a(3),b(3);
	// a << 1,2,3;
	// b << 3,2,1;
	// cout << a << endl;
	// cout << a.dot(b) << endl;
	// MatrixXd m(1,5);
	// m << 1,2,3,4,5;
	// cout << m <<endl<<endl;
	// m.conservativeResize(2,5);
	// m.bottomRows(1) << 2,3,4,5,6;
	// cout << m << endl;
	// MatrixXf matA(4, 1);
	// matA << 1, 2, 3, 4;
	// cout << -matA <<endl;
	// MatrixXf matB(4, 4);
	// matB.col(0) << matA;
	// matB.col(1) << matA;
	// matB.col(2) << matA;
	// matB.col(3) << matA;
	// cout << matB << endl;
	// VectorXd a(3),b(3);
	// VectorXd c;
	// c = VectorXd::Zero(3);
	// a << 1,2,3;
	// b << 2,3,4;
	// cout << c <<endl;
	// cout << (a+b).mean() << endl; //failed
	// a = 1-a.array();
	// cout << a.cwiseProduct(b) << endl; //works
}
	// cout << 1-b.array() << endl;
	// b = 1-b.array();
	// cout << b.cwiseProduct(a) << endl;
	// cout << (MatrixXd::Ones(b.rows(),1)-b).cwiseProduct(a) << endl;
	// VectorXd sigmoid;
	// sigmoid << 1,2,3;
	// VectorXd siglog = sigmoid.array().log();
	// cout << (VectorXd::Ones(siglog.rows())-sigmoid).array().log() << endl;
	// int array[8];
	// for(int i = 0; i < 8; ++i) array[i] = i;
	// Matrix<float,2,4> data;
	// std::cout << "Column-major:\n" << Eigen::Map<Eigen::Matrix<int,2,4>>(array) << std::endl;
	// double i;
 //  MatrixXd m(2,2);
 //  MatrixXd n(2,1);
 //  m(0,0) = 3;
 //  m(1,0) = 2.5;
 //  m(0,1) = -1;
 //  m(1,1) = m(1,0) + m(0,1);
 //  m = m.array() + 1 ;
  // n = m.rowwise().mean();
  // i = n.array().sum();
  // std::cout << m << std::endl;
  // std::cout << i << std::endl;
