#include "include/covtype.h"
using namespace Eigen;
typedef Triplet<double> T;

// const int trainSetSize = 400000;
// const int testSetSize = 180000;
const int trainTripletEstimateSize = 2000000;
// const int testTripletEstimateSize = 1000000;

inline void tripletInit(std::ifstream &file, std::string &line,
                        std::vector<T> &triplet, std::string &slabel,
                        int &label, std::string &srow, int &row,
                        std::string &svalue, int &value, std::string &element,
                        std::vector<double> &labelList, const int &datasetSize,
                        int &col) {
  int count = 1;
  while (count <= datasetSize && getline(file, line)) {
    std::stringstream linestream(line);
    // get label of each line
    if (linestream.good()) {
      getline(linestream, slabel, ' ');
      label = atof(slabel.c_str());
      // std::cout << "label: " << label << std::endl;
      // chage label to 0|1
      labelList.push_back(label - 1);
      while (linestream.good()) {
        getline(linestream, element, ' ');
        std::stringstream elementstream(element);
        getline(elementstream, srow, ':');
        getline(elementstream, svalue, '.');
        row = atof(srow.c_str()) - 1;  // change range from 1-54 to 0-53
        value = atof(svalue.c_str());
        triplet.push_back(T(row, col, value));
        // std::cout << "row: " << row << " value: " << value << std::endl;
      }
    }
    ++count;
    ++col;
  }
  return;
}

void covtype_binary_read(SparseMatrix<double> &Xt, VectorXd &y, int setSize,
                         std::string full_path) {
  MatrixXd Xt, XtTest;
  std::vector<T> trainTripletList;
  std::vector<double> trainLabelList;
  trainTripletList.reserve(trainTripletEstimateSize);
  std::ifstream file;
  int label, row = 0, col = 0, value;
  std::vector<double> fulldata;
  std::string line, slabel, srow, svalue, element;
  SPARSE = 1;
  file.open(full_path);
  if (file.is_open()) {
    tripletInit(file, line, trainTripletList, slabel, label, srow, row, svalue,
                value, element, trainLabelList, setSize, col);
  }

  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> yy(trainLabelList.data(),
                                                     setSize, 1);
  y = yy;
  std::cout << Xt.rows() << " " << Xt.cols() << std::endl;
  Xt.setFromTriplets(trainTripletList.begin(), trainTripletList.end());
  // normalization
  for (int k = 0; k < Xt.outerSize(); ++k) {
    int colNorm = Xt.col(k).norm();
    Xt.col(k) /= colNorm;
  }
  XtS = Xt.sparseView();
  XtTestS = XtTest.sparseView();
  file.close();
  return;
}