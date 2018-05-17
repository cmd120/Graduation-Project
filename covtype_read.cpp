#include "include/covtype.h"
using namespace Eigen;
void covtype_read(MatrixXd &Xt, VectorXd &y, MatrixXd &XtTest,
                  VectorXd &yTest) {
  std::ifstream file;
  std::string line, num;
  std::string full_path = "covtype.data";
  std::vector<double> fulldata, covtype_dataset, covtype_labels;
  fulldata.reserve(40000000);
  file.open(full_path);
  if (file.is_open()) {
    while (getline(file, line)) {
      std::istringstream linestream(line);
      while (linestream.good()) {
        getline(linestream, num, ',');
        fulldata.push_back(atof(num.c_str()));
      }
    }

  } else {
    // administor required
    // throw runtime_error("Cannot open file `" + full_path + "`!");
    ;
  }
  SPARSE = issparse(fulldata);
  for (int i = 0; i < fulldata.size() / 55; ++i) {
    int label = fulldata[i * 55 + 54];
    // binary classification
    if (label == 1 || label == 2) {
      covtype_dataset.insert(covtype_dataset.end(), fulldata.begin() + i * 55,
                             fulldata.begin() + i * 55 + 54);
      covtype_labels.push_back(label-1);
    }
  }
  std::cout << "label set size:" << covtype_labels.size() << std::endl;
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> Xtt(
      covtype_dataset.data(), 54, covtype_dataset.size() / 54);
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> yy(covtype_labels.data(),
                                                     covtype_labels.size(), 1);
  // std::cout << "Xtt[0]: " << Xtt.col(0) << std::endl;
  Xt = Xtt.leftCols(trainSetSize), y = yy.topRows(trainSetSize),
  XtTest = Xtt.rightCols(testSetSize), yTest = yy.bottomRows(testSetSize);
  for (int i = 0; i < Xt.cols(); ++i) {
    Xt.col(i) /= Xt.col(i).norm();
  }
  return;
}

// int main(){
// 	MatrixXd Xt, XtTest;
// 	VectorXd y, yTest;
// 	covtype_read(Xt, y, XtTest, yTest);
// 	// std::cout << Xt.cols() << std::endl;
// 	return 0;
// }