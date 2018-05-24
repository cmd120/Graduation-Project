#ifndef DATA_H
#define DATA_H
#include <fstream>
#include <iterator>
#include <sstream>
#include "comm.h"
typedef unsigned char BYTE;
typedef Eigen::Triplet<double> T;  ///< A triplet.
const int TripletEstimateSize = 2000000;
const int DatasetEstimateSize = 40000000;
const int trainSetSize = 100;
const int testSetSize = 20;
///
/// Deals with integer difference due to different endian between processors.
///
auto reverseInt = [](int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};
///
/// Remove the leading and trailing spaces from a string.
///
std::string trim(const std::string &str,
                 const std::string &whitespace = " \t");
///
/// Reduce the extra spaces inside a string base on trim().
///
std::string reduce(const std::string &str, const std::string &fill = " ",
                   const std::string &whitespace = " \t");
/// Build the tripletlist to initialize a sparse matrix.
/// A triplet is a small structure to hold a non zero, here represented as
/// (row,col,value). This funciton takes in a sparse format datafile and works
/// on the assumption that label value of data is -1,1 or 1,2.
///
void tripletInit(std::ifstream &file, std::string &line,
                        std::vector<T> &triplet, std::string &slabel,
                        double &label, std::string &srow, std::string &svalue,
                        double &value, std::string &element,
                        std::vector<double> &labelList,
                        const int &datasetSize);
///
/// Read in the image file of mnist dataset and return the corresponding data
/// vector.
///
std::vector<BYTE> read_mnist_images(std::string full_path);
///
/// Read in the label file of mnist dataset and return the corresponding data
/// vector.
///
std::vector<BYTE> read_mnist_labels(std::string full_path);
///
/// Read the MNIST handwritten digits dataset, proccessed for binary
/// classification. Also be a example of writing custom dataset processing
/// function.
///
void mnist_read(Eigen::MatrixXd &Xt, Eigen::VectorXd &y,
                Eigen::MatrixXd &XtTest, Eigen::VectorXd &yTest);
///
/// Read Dense Format Dataset File.
///
void DenseFormatRead(Eigen::MatrixXd &Xt, Eigen::VectorXd &y, int features,
                     int nSamples, std::string filename);
///
/// Read Sparse Format Dataset File.
///
void SparseFormatRead(Eigen::SparseMatrix<double> &XtS, Eigen::VectorXd &y,
                      int features, int nSamples, std::string filename);

#endif