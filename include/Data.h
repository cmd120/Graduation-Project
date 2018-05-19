#ifndef DATA_H
#define DATA_H
#include <fstream>
#include <iterator>
#include <sstream>
#include "comm.h"
typedef unsigned char BYTE;
const int TripletEstimateSize = 2000000;
const int DatasetEstimateSize = 40000000;
const int trainSetSize = 100;
const int testSetSize = 20;
//*********MNIST*********//
auto reverseInt = [](int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

std::vector<BYTE> read_mnist_images(std::string full_path);
std::vector<BYTE> read_mnist_labels(std::string full_path);
void mnist_read(Eigen::MatrixXd &Xt, Eigen::VectorXd &y,
                Eigen::MatrixXd &XtTest, Eigen::VectorXd &yTest);

void DenseFormatRead(Eigen::MatrixXd &Xt, Eigen::VectorXd &y, int features,
                     int nSamples, std::string filename);
// General read function for sparse-format file
void SparseFormatRead(Eigen::SparseMatrix<double> &XtS, Eigen::VectorXd &y,
                      int features, int nSamples, std::string filename);

#endif