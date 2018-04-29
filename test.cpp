#include "include/comm.h"
#include "include/DenseMat.h"
#include "include/SparseMat.h"
#include "include/MNIST_Read.h"
#include "include/IAG.h"
#include "include/IAGA.h"
#include "include/SGD.h"
#include "include/SAG.h"
#include "include/SIG.h"
#include "include/SVRG"

void removeRow(MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

void removeColumn(MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}
int main()
{	
	int i;
    string train_image_path = "train-images";
    string train_label_path = "train-labels";
    string test_image_path = "test-images";
    string test_label_path = "test-labels";
    vector<BYTE> train_image_dataset = read_mnist_images(train_image_path);
    vector<BYTE> train_label_dataset = read_mnist_labels(train_label_path);
 	vector<BYTE> test_image_dataset = read_mnist_images(test_image_path);
    vector<BYTE> test_label_dataset = read_mnist_labels(test_label_path);
    // cout << "SPARSE: " << IsSparseMat<BYTE>(test_image_dataset) << endl;
    // cout << "SPARSE: " << endl;
    // MatrixXd Mdata(_dataset.data(),28,_dataset.size()/28);
    vector<double> Xt_train(train_image_dataset.begin(),train_image_dataset.end());
    vector<double> y_train(train_label_dataset.begin(),train_label_dataset.end());
    vector<double> Xt_test(test_image_dataset.begin(),test_image_dataset.end());
    vector<double> y_test(test_label_dataset.begin(),test_label_dataset.end());
    
    vector<double> Xt_train_classify,y_train_classify,Xt_test_classify,y_test_classify;
    for(i=0;i<y_train.size();++i){
    	if(y_train[i]<=1){
    		Xt_train_classify.insert(Xt_train_classify.end(),Xt_train.begin()+i*784,Xt_train.begin()+(i+1)*784);
    		y_train_classify.push_back(y_train[i]);
    	}
    }
    for(i=0;i<y_test.size();++i){
    	if(y_test[i]<=1){
    		Xt_test_classify.insert(Xt_test_classify.end(),Xt_test.begin()+i*784,Xt_test.begin()+(i+1)*784);
    		y_test_classify.push_back(y_test[i]);
    	}
    }
    // Map<Matrix<double,Dynamic,Dynamic,ColMajor>> Xt(Xt_train.data(), 784, Xt_train.size()/784);
    // Map<Matrix<double,Dynamic,Dynamic,ColMajor>> y(y_train.data(), y_train.size(), 1);
    // Map<Matrix<double,Dynamic,Dynamic,ColMajor>> XtTest(Xt_test.data(), 784, Xt_test.size()/784);
    // Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yTest(y_test.data(), y_test.size(), 1);
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> Xt(Xt_train_classify.data(), 784, Xt_train_classify.size()/784);
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> y(y_train_classify.data(), y_train_classify.size(), 1);
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> XtTest(Xt_test_classify.data(), 784, Xt_test_classify.size()/784);
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yTest(y_test_classify.data(), y_test_classify.size(), 1);
	//normalization
    for(i=0;i<Xt.cols();++i){
    	Xt.col(i) = Xt.col(i)/Xt.col(i).norm();
    }
    for(i=0;i<XtTest.size()/784;++i){
    	XtTest.col(i) = XtTest.col(i)/XtTest.col(i).norm();
    }
    VectorXd yy = y;
    VectorXd yyTest = yTest;
    VectorXd w,sumIG(Xt.rows()),gradients(Xt.cols());
    w = MatrixXd::Zero(Xt.rows(),1);
    w = VectorXd(w);
    gradients = (1+(-Xt.adjoint()*w).array().exp()).inverse() - y.array();
    sumIG = Xt*gradients;
    string filename = "dong";
    cout << "enter IAG" << endl;
    cout << "Xt rows: " << Xt.rows() << "Xt cols: " << Xt.cols() << endl;
    cout << "XtTest rows: " << XtTest.rows() << "XtTest cols: " << XtTest.cols() << endl;
    cout << "y cols: " << yy.size() << endl;
    cout << "yTest cols: " << yyTest.size() << endl;
    IAG_logistic(w, Xt, yy, XtTest, yyTest, sumIG, gradients, filename);
    IAGA_logistic(w, Xt, yy, XtTest, yyTest, sumIG, gradients, filename);
    SIG_logistic(w, Xt, yy, XtTest, yyTest, sumIG, gradients, filename);
    SAG_logistic(w, Xt, yy, XtTest, yyTest, sumIG, gradients, filename);
    SGD_logistic(w, Xt, yy, XtTest, yyTest, sumIG, gradients, filename);
    SVRG_logistic(w, Xt, yy, XtTest, yyTest, sumIG, gradients, filename);
    return 0;
    // cout << Xt.col(0) << endl;
    
    // VectorXd a(3),b(3);
    // a(0) = 1;
    // a(1) = 2;
    // a(2) = 3;
    // b(0) = 1;
    // b(1) = 2;
    // b(2) = 3;
    // a-b;
    // return 0;
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
