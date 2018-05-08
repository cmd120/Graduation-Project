#include "include/covtype.h"

typedef Triplet<double> T;

// const int trainSetSize = 400000;
// const int testSetSize = 180000;
const int trainTripletEstimateSize = 2000000;
// const int testTripletEstimateSize = 1000000;

inline void tripletInit(ifstream &file, string &line, vector<T> &triplet, string &slabel, int &label,\
						 string &srow, int &row, string &svalue, int &value,\
						 string &element, vector<double> &labelList, const int &datasetSize, int &col){
	int count = 1;
	while(count<=datasetSize&&getline(file,line)){
		istringstream linestream(line);
		//get label of each line
		if(linestream.good()){
			getline(linestream,slabel,' ');
			label = atof(slabel.c_str());
			// cout << "label: " << label << endl;
			//chage label to 0|1
			labelList.push_back(label-1);
			while(linestream.good()){
				getline(linestream, element, ' ');
				istringstream elementstream(element);
				getline(elementstream,srow,':');
				getline(elementstream,svalue,'.');
				row = atof(srow.c_str()) - 1;//change range from 1-54 to 0-53
				value = atof(svalue.c_str());
				triplet.push_back(T(row,col,value));
				// cout << "row: " << row << " value: " << value << endl;
			}
		}
		++count;
		++col;
	}
	return;
}

void covtype_binary_read(SparseMatrix<double> &Xt, VectorXd &y, int setSize, string full_path/*, SparseMatrix<double> &XtTest, VectorXd &yTest*/){
	vector<T> trainTripletList/*,testTripletList*/;
	vector<double> trainLabelList/*,testLabelList*/;
	trainTripletList.reserve(trainTripletEstimateSize);
	// testTripletList.reserve(testTripletEstimateSize);
	ifstream file;
	int label, row=0, col=0, value;
	vector<double> fulldata;
	string line, slabel, srow, svalue, element;
	// string full_path = "cov.test";
	// string full_path = "covtype.libsvm.binary";
	SPARSE = 1;
	file.open(full_path);
	if(file.is_open()){
		tripletInit(file, line, trainTripletList, slabel, label, srow, row, svalue, value, element, trainLabelList, setSize, col);
		// tripletInit(file, line, testTripletList, slabel, label, srow, row, svalue, value, element, testLabelList, testSetSize, col);
	}
	cout << "train triplet list size: " << trainTripletList.size() << endl;
	// cout << "test triplet list size: " << testTripletList.size() << endl;
	cout << "train triplet label size: " << trainLabelList.size() << endl;
	// cout << "test triplet label size: " << testLabelList.size() << endl;
	Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yy(trainLabelList.data(), setSize, 1);
	// Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yyTest(testLabelList.data(), testSetSize, 1);
	y = yy;/* yTest = yyTest;*/
	// cout << "pass 1" << endl;
	cout << Xt.rows() << " " << Xt.cols() << endl;
	Xt.setFromTriplets(trainTripletList.begin(),trainTripletList.end());
	//normalization
	for(int k=0;k<Xt.outerSize();++k){
		int colNorm = Xt.col(k).norm();
		Xt.col(k)/=colNorm;
	}

	// cout << "pass 2"  << endl;
	// cout << XtTest.rows() << " " << XtTest.cols() <<endl;
// //**********************************error here, unsolved*****************************************
// 	XtTest.setFromTriplets(testTripletList.begin(),testTripletList.end());
// 	cout << "pass 3" << endl;
// //
	file.close();
	return;
}
// int main(){
// 	string full_path = "covtype.libsvm.binary";
// 	SparseMatrix<double> mat(54,trainSetSize);
// 	SparseMatrix<double> matTest(54,testSetSize);
// 	VectorXd y,yTest;
// 	covtype_binary_read(full_path,mat,y, trainSetSize);
// 	covtype_binary_read(full_path,matTest,yTest, testSetSize);
// 	return 0;
// }