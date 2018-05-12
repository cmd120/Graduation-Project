#ifndef COMM_H
#define COMM_H

#define NOISY 0  // 0 for optimization, 1 for bayes model
#define PRINT_FREQ 50 // print test error 50 times per epoch
#define BILLION  1E9
#define FILE_NAME_LENGTH 64
#define ACCURACY 10E-5
//#define EIGEN_USE_MKL_ALL //if Intel MKL is installed

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "Noise.h"
#include "LogisticError.h"
#include "LogisticGradient.h"




extern int epochCounter;
extern FILE *fp;
extern std::chrono::high_resolution_clock::time_point startTime;
extern int SPARSE;
//Windows timer resolution
#if defined(_WIN32) || defined(_WIN64)
using Clock = std::chrono::high_resolution_clock;
#endif
//Linux/Unix timer resolution
#if defined(__linux__) || defined(__unix) || defined(__unix__)
// temporary
using Clock = std::chrono::high_resolution_clock;
//when extern variables placed here, endif seems not match to the previous unmatched one but closest one??
#endif
// void LogisticGradient(double *w, const mxArray *XtArray, double *y, double *G);

// void RidgeError(double *w, const mxArray *XtArray, double *y, double epoch, double telapsed, FILE *fp);

// void RidgeGradient(double *w, const mxArray *XtArray, double *y, double *G);

// void Shuffle(int *data, int num);

// double NoiseGen(double mean, double variance);

#endif
//TODOS：
//使用引用传参避免复制，考虑const的问题