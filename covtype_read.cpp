#include "include/covtype.h"

void covtype_read(MatrixXd &Xt, VectorXd &y, MatrixXd &XtTest, VectorXd &yTest){
	ifstream file;
	string line, num;
	string full_path = "covtype.data";
	vector<double> fulldata, covtype_dataset, covtype_labels;
	fulldata.reserve(40000000);
	file.open(full_path);
	if(file.is_open()){
		while(getline(file,line)){
			istringstream linestream(line);
			while(linestream.good()){
				getline(linestream,num,',');
				fulldata.push_back(atof(num.c_str()));
			}
		}

	}else{
		//administor required
		// throw runtime_error("Cannot open file `" + full_path + "`!");
		;
	}
	SPARSE = issparse(fulldata);
	for(int i=0;i<fulldata.size()/55;++i){
		int label = fulldata[i*55+54];
		//binary classification
		if(label==1||label==0){
			covtype_dataset.insert(covtype_dataset.end(),fulldata.begin()+i*55,fulldata.begin()+i*55+54);
			covtype_labels.push_back(label);
		}
	}
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> Xtt(covtype_dataset.data(), 54, covtype_dataset.size()/54);
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yy(covtype_labels.data(), covtype_labels.size(), 1);
	// cout << "Xtt[0]: " << Xtt.col(0) << endl;
	Xt = Xtt.leftCols(trainSetSize), y = yy.topRows(trainSetSize), XtTest = Xtt.rightCols(testSetSize), yTest = yy.bottomRows(testSetSize);
	for(int i=0;i<Xt.cols();++i){
		Xt.col(i) /= Xt.col(i).norm();
	}
	return;
}

// int main(){
// 	MatrixXd Xt, XtTest;
// 	VectorXd y, yTest;
// 	covtype_read(Xt, y, XtTest, yTest);
// 	// cout << Xt.cols() << endl;
// 	return 0;
// }