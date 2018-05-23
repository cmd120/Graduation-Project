#include "include/Data.h"
using namespace Eigen;
typedef Triplet<double> T;

//*********TRIPLETINIT*********//
std::string trim(const std::string &str,
                 const std::string &whitespace = " \t") {
  const auto strBegin = str.find_first_not_of(whitespace);
  if (strBegin == std::string::npos) return "";  // no content

  const auto strEnd = str.find_last_not_of(whitespace);
  const auto strRange = strEnd - strBegin + 1;

  return str.substr(strBegin, strRange);
}
std::string reduce(const std::string& str,
                   const std::string& fill = " ",
                   const std::string& whitespace = " \t")
{
    // trim first
    auto result = trim(str, whitespace);

    // replace sub ranges
    auto beginSpace = result.find_first_of(whitespace);
    while (beginSpace != std::string::npos)
    {
        const auto endSpace = result.find_first_not_of(whitespace, beginSpace);
        const auto range = endSpace - beginSpace;

        result.replace(beginSpace, range, fill);

        const auto newStart = beginSpace + fill.length();
        beginSpace = result.find_first_of(whitespace, newStart);
    }

    return result;
}
inline void tripletInit(std::ifstream &file, std::string &line,
                        std::vector<T> &triplet, std::string &slabel,
                        double &label, std::string &srow, std::string &svalue,
                        double &value, std::string &element,
                        std::vector<double> &labelList,
                        const int &datasetSize) {
  int row, col = 0;
  while (col < datasetSize && getline(file, line)) {
    line = reduce(line);
    // std::cout << line+"####" << std::endl;
    // return;
    std::stringstream linestream(line);
    // Get label of each line
    if (linestream.good()) {
      getline(linestream, slabel, ' ');
      label = atof(slabel.c_str());
      // Default: Deal with binary classification problem
      // Except covtype dataset labels(1 or 2), others labels are -1 or 1
      if (abs((int)label) != 1 && abs((int)label) != 2)
        continue;
      else {
        label = (int)label == -1 ? (label + 1) : label;
        std::cout << "label:" << label << std::endl;
      }
      // Chage label range to 0|1
      (int)label > 1 ? labelList.push_back(label - 1)
                     : labelList.push_back(label);
      while (linestream.good()) {
        getline(linestream, element, ' ');
        std::stringstream elementstream(element);
        getline(elementstream, srow, ':');
        getline(elementstream, svalue);
        row = atof(srow.c_str()) -
              1;  // change range from 1|features to 0|(features-1)
        value = atof(svalue.c_str());
        printf("row:%d,col:%d,value:%f\n", row, col, value);
        triplet.push_back(T(row, col, value));
      }
      //DEBUG return;
      // return;
    }
    ++col;
  }
  return;
}

//*********MINIST*********//
std::vector<BYTE> read_mnist_images(std::string filename) {
  std::string full_path = "data/" + filename;
  int number_of_images, image_size;  // modification
  auto reverseInt = [](int i) {
    BYTE c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255,
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  };

  std::ifstream file(full_path, std::ios::binary);

  if (file.is_open()) {
    int magic_number = 0, n_rows = 0, n_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    if (magic_number != 2051)
      throw std::runtime_error("Invalid MNIST image file!");
    file.read((char *)&number_of_images, sizeof(number_of_images)),
        number_of_images = reverseInt(number_of_images);
    file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);
    std::istreambuf_iterator<char> start(file), end;
    std::vector<BYTE> dataset(start, end);
    return dataset;
  } else {
    throw std::runtime_error("Cannot open file `" + full_path + "`!");
  }
}

std::vector<BYTE> read_mnist_labels(std::string filename) {
  std::string full_path = "data/" + filename;
  int number_of_labels;
  std::ifstream file(full_path, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    if (magic_number != 2049)
      throw std::runtime_error("Invalid MNIST label file!");
    file.read((char *)&number_of_labels, sizeof(number_of_labels)),
        number_of_labels = reverseInt(number_of_labels);
    std::istreambuf_iterator<char> start(file), end;
    std::vector<BYTE> dataset(start, end);
    return dataset;
  } else {
    throw std::runtime_error("Unable to open file `" + full_path + "`!");
  }
}

void mnist_read(MatrixXd &Xt, VectorXd &y, MatrixXd &XtTest, VectorXd &yTest) {
  const int features = 784;
  std::string train_image_name = "mnist.train.images";
  std::string train_label_name = "mnist.train.labels";
  std::string test_image_name = "mnist.test.images";
  std::string test_label_name = "mnist.test.labels";
  std::vector<BYTE> train_image_dataset = read_mnist_images(train_image_name);
  std::vector<BYTE> train_label_dataset = read_mnist_labels(train_label_name);
  std::vector<BYTE> test_image_dataset = read_mnist_images(test_image_name);
  std::vector<BYTE> test_label_dataset = read_mnist_labels(test_label_name);

  std::vector<double> Xt_train(train_image_dataset.begin(),
                               train_image_dataset.end());
  std::vector<double> y_train(train_label_dataset.begin(),
                              train_label_dataset.end());
  std::vector<double> Xt_test(test_image_dataset.begin(),
                              test_image_dataset.end());
  std::vector<double> y_test(test_label_dataset.begin(),
                             test_label_dataset.end());
  // classification
  std::vector<double> Xt_train_classify, y_train_classify, Xt_test_classify,
      y_test_classify;
  for (int i = 0; i < y_train.size(); ++i) {
    if (y_train[i] <= 1) {
      Xt_train_classify.insert(Xt_train_classify.end(),
                               Xt_train.begin() + i * features,
                               Xt_train.begin() + (i + 1) * features);
      y_train_classify.push_back(y_train[i]);
    }
  }
  for (int i = 0; i < y_test.size(); ++i) {
    if (y_test[i] <= 1) {
      Xt_test_classify.insert(Xt_test_classify.end(),
                              Xt_test.begin() + i * features,
                              Xt_test.begin() + (i + 1) * features);
      y_test_classify.push_back(y_test[i]);
    }
  }
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> Xtt(
      Xt_train_classify.data(), features, Xt_train_classify.size() / features);
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> yy(
      y_train_classify.data(), y_train_classify.size(), 1);
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> XttTest(
      Xt_test_classify.data(), features, Xt_test_classify.size() / features);
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> yyTest(
      y_test_classify.data(), y_test_classify.size(), 1);
  // normalization
  for (int i = 0; i < Xtt.cols(); ++i) {
    Xtt.col(i) = Xtt.col(i) / Xtt.col(i).norm();
  }
  for (int i = 0; i < XttTest.cols(); ++i) {
    XttTest.col(i) = XttTest.col(i) / XttTest.col(i).norm();
  }
  Xt = Xtt;
  y = yy;
  XtTest = XttTest;
  yTest = yyTest;
}

// Read Dense Format Dataset File
void DenseFormatRead(MatrixXd &Xt, VectorXd &y, int features, int nSamples,
                     std::string filename) {
  std::ifstream file;
  std::string line, num;
  std::string full_path = "data/" + filename;
  std::vector<double> fulldata, covtype_dataset, covtype_labels;
  fulldata.reserve(DatasetEstimateSize);
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
    throw std::runtime_error("Cannot open file `" + full_path + "`!");
  }
  SPARSE = issparse(fulldata);
  for (int i = 0; i < fulldata.size() / (features + 1); ++i) {
    int label = fulldata[i * (features + 1) + features];
    // binary classification
    if (label == 1 || label == 2) {
      covtype_dataset.insert(covtype_dataset.end(),
                             fulldata.begin() + i * (features + 1),
                             fulldata.begin() + i * (features + 1) + features);
      covtype_labels.push_back(label - 1);
    }
  }
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> Xtt(
      covtype_dataset.data(), features, covtype_dataset.size() / features);
  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> yy(covtype_labels.data(),
                                                     covtype_labels.size(), 1);
  // design your train/test dataset
  Xt = Xtt.leftCols(nSamples), y = yy.topRows(nSamples);
  // normalization
  for (int i = 0; i < Xt.cols(); ++i) {
    Xt.col(i) /= Xt.col(i).norm();
  }
  return;
}
// Read Sparse Format Dataset File
void SparseFormatRead(SparseMatrix<double> &XtS, VectorXd &y, int features,
                      int nSamples, std::string filename) {
  SparseMatrix<double> tmp(features, nSamples);
  XtS = tmp;
  std::vector<T> TripletList;
  std::vector<double> LabelList;
  TripletList.reserve(TripletEstimateSize);
  std::ifstream file;
  double label, value;
  std::vector<double> fulldata;
  std::string line, slabel, srow, svalue, element;
  std::string full_path = "data/" + filename;
  SPARSE = 1;
  file.open(full_path);
  if (file.is_open()) {
    tripletInit(file, line, TripletList, slabel, label, srow, svalue, value,
                element, LabelList, nSamples);
  }

  Map<Matrix<double, Dynamic, Dynamic, ColMajor>> yy(LabelList.data(), nSamples,
                                                     1);
  y = yy;
  std::cout << "enter SparseMatrix init" << std::endl;
  XtS.setFromTriplets(TripletList.begin(), TripletList.end());
  std::cout << "XtS.cols(): " << XtS.cols() << std::endl;
  // normalization
  for (int k = 0; k < XtS.outerSize(); ++k) {
    int colNorm = XtS.col(k).norm();
    XtS.col(k) /= colNorm;
  }
  file.close();
  return;
}
