#ifndef COMM_H
#define COMM_H

#include <cstdio>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "Noise.h"


#define NOISY 1  // 0 for optimization, 1 for bayes model
#define PRINT_FREQ 50 // print test error 10 times per epoch
#define BILLION  1E9
#define FILE_NAME_LENGTH 64
#define ACCURACY 10E-5


// void LogisticGradient(double *w, const mxArray *XtArray, double *y, double *G);

// void RidgeError(double *w, const mxArray *XtArray, double *y, double epoch, double telapsed, FILE *fp);

// void RidgeGradient(double *w, const mxArray *XtArray, double *y, double *G);

// void Shuffle(int *data, int num);

// double NoiseGen(double mean, double variance);

#endif
//TODOS：
//使用引用传参避免复制，考虑const的问题