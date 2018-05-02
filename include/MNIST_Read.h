#ifndef MINIST_READ_H
#define MINIST_READ_H

#include <fstream>
#include <iterator>

#include "comm.h"

using namespace std;

typedef unsigned char BYTE;
//reverse int
auto reverseInt = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

vector<BYTE> read_mnist_images(string full_path);
vector<BYTE> read_mnist_labels(string full_path);
void mnist_read(MatrixXd &Xt, VectorXd &y, MatrixXd &XtTest, VectorXd &yTest);

#endif
