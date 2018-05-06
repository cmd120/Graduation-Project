#include "include/covtype.h"

void covtype_read(MatrixXd &Xt, VectorXd &y, MatrixXd &XtTest, VectorXd &yTest){
	ifstream file;
	string line, num;
	string full_path = "covtype.data";
	vector<double> fulldata, covtype_dataset, covtype_labels;
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
	// SPARSE = issparse(fulldata);
	for(int i=0;i<fulldata.size()/55;++i){
		covtype_dataset.insert(covtype_dataset.end(),fulldata.begin()+i*55,fulldata.begin()+i*55+54);
		covtype_labels.push_back(fulldata[i*55+54]);
	}
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> Xtt(covtype_dataset.data(), 54, covtype_dataset.size()/54);
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yy(covtype_labels.data(), covtype_labels.size(), 1);
	int train_cols(389278), test_cols(191734);
	// cout << train_cols << " " << test_cols;
	Xt = Xtt.leftCols(train_cols), y = yy.topRows(train_cols), XtTest = Xtt.rightCols(test_cols), yTest = yy.bottomRows(test_cols);
	return;
}

// int main(){
// 	MatrixXd Xt, XtTest;
// 	VectorXd y, yTest;
// 	covtype_read(Xt, y, XtTest, yTest);
// 	return 0;
// }