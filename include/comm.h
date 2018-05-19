#ifndef COMM_H
#define COMM_H

#define DEBUG 1
#define NOISY 0        // 0 for optimization, 1 for bayes model
#define PRINT_FREQ 50  // print test error 50 times per epoch
#define BILLION 1E9
#define FILE_NAME_LENGTH 64
#define ACCURACY 10E-5
//#define EIGEN_USE_MKL_ALL //if Intel MKL is installed
typedef enum {
  DEFAULT = 0,
  SUCCESS,
  STEPLENGTHTOOBIG,
  STEPLENGTHTOOSMALL,
  ALLOCERROR,
} ERRORCODE;

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "DenseMat.h"
#include "LogisticGradient.h"
#include "Noise.h"
#include "SparseMat.h"

class LR {
 private:
  double lambda = 0;
  double L = 0;
  double mu = 0;
  double optSolution;
  double optConst;
  void sigfunc(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y,
               Eigen::VectorXd &sigmoid) {
    sigmoid = (1 + (-X.adjoint() * w).array().exp()).cwiseInverse();
  }

 public:
  LR(double lambda = 0, double L = 0, double mu = 0) {
    this->lambda = lambda;
    this->L = L;
    this->mu = mu;
  }
  double costfunc(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y) {
    Eigen::VectorXd sigmoid;
    sigfunc(w, X, y, sigmoid);
    Eigen::VectorXd siglog = sigmoid.array().log();
    Eigen::VectorXd siglogimpl = (1 - sigmoid.array()).log();
    // mean maybe wrong here
    double loss =
        -(y.dot(siglog) + (1 - y.array()) * siglogimpl.array()).mean();
    double regularizer = this->lambda / 2 * (w.array().square()).sum();
    return loss + regularizer;
  }
  double loglikelihood(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                       Eigen::VectorXd &y) {
    Eigen::VectorXd sigmoid;
    sigfunc(w, X, y, sigmoid);
    // mean may be wrong here
    double loglikelihood = (y.array() * sigmoid.array().log() +
                            (1 - y.array()) * (1 - sigmoid.array()).log())
                               .mean();
    return loglikelihood;
  }
  Eigen::VectorXd grad_noreg(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                             Eigen::VectorXd &y) {
    int d = X.rows(), n = X.cols();
    Eigen::VectorXd tmpExp = (-(X.adjoint() * w)).array().exp();
    Eigen::VectorXd gradient =
        X * (1 + tmpExp.array() - y.array()).matrix().cwiseInverse() / n;
    return gradient;
  }
  void hypothesis(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                  Eigen::VectorXd &hypothesis) {
    hypothesis = 1 / (1 + (-X.adjoint() * w).array().exp().array());
  }
  void costprint(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y,
                 int stage) {
    double cost = costfunc(w, X, y);
    double grad_square = (grad_noreg(w, X, y) + lambda * w).norm();
    // fprintf('epoch: %4d, cost: %.25f, grad: %.25f\n', stage, cost,
    // grad_square);
    // fprintf('epoch: %4d, cost: %.25f\n', stage, cost);  // resize the cost
  }
  void predictfunc(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                   Eigen::VectorXd &labels) {
    int n = X.cols();
    Eigen::VectorXd y;
    hypothesis(w, X, y);
    labels = Eigen::VectorXd::Ones(n);
    for (int i = 0; i < n; i++) {
      if (y(i) < 0.5) labels(i) = 0;
    }
  }
  double score(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y) {
    int n = X.cols();
    double score = 0;
    Eigen::VectorXd labels;
    predictfunc(w, X, labels);
    for (int i = 0; i < n; i++) {
      if (labels[i] == y[i]) score += 1;
    }
    if (DEBUG) {
      std::cout << "score: " << score << std::endl;
      std::cout << "n: " << n << std::endl;
    }
    return score / n;
  }
  ~LR() { ; };
};
class RR {
 private:
  double lambda = 0;
  double L = 0;
  double mu = 0;
  double optSolution;
  double optConst;

 public:
  RR(double lambda = 0, double L = 0, double mu = 0) {
    this->lambda = lambda;
    this->L = L;
    this->mu = mu;
  }
  double costfunc(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y) {
    Eigen::VectorXd sigmoid;
    double loss = (y - X.adjoint() * w).array().square().mean() / 2;
    double regularizer = this->lambda / 2 * (w.array().square()).sum();
    return loss + regularizer;
  }
  double loglikelihood(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                       Eigen::VectorXd &y) {
    Eigen::VectorXd innerProd;
    innerProd = X.adjoint() * w;
    double loglikelihood = 1 / 2 * (innerProd - y).array().square().mean();
    return loglikelihood;
  }
  Eigen::VectorXd grad_noreg(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                             Eigen::VectorXd &y) {
    int d = X.rows(), n = X.cols();
    Eigen::VectorXd gradient = 1 / n * (X * (X.adjoint() * w - y));
  }
  void hypothesis(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                  Eigen::VectorXd &hypothesis) {
    hypothesis = X.adjoint() * w;
  }
  void costprint(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y,
                 int stage) {
    double cost = costfunc(w, X, y);
    double grad_square = (grad_noreg(w, X, y) + this->lambda * w).norm();
    // fprintf('epoch: %4d, cost: %.25f, grad: %.25f\n', stage, cost,
    // grad_square);
    // fprintf('epoch: %4d, cost: %.25f\n', stage, cost);  // resize the cost
  }
  void predictfunc(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                   Eigen::VectorXd &labels) {
    int n = X.cols();
    Eigen::VectorXd y;
    hypothesis(w, X, y);
    labels = Eigen::VectorXd::Ones(n);
    for (int i = 0; i < n; i++) {
      if (y(i) < 0) labels(i) = 0;
    }
  }
  double score(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y) {
    int n = X.cols();
    double score = 0;
    Eigen::VectorXd labels;
    predictfunc(w, X, labels);
    for (int i = 0; i < n; i++) {
      if (labels[i] == y[i]) score += 1;
    }
    if (DEBUG) {
      std::cout << "score: " << score << std::endl;
      std::cout << "n: " << n << std::endl;
    }
    return score / n;
  }
  ~RR() { ; };
};
extern int epochCounter;
extern FILE *fp;
extern std::chrono::high_resolution_clock::time_point startTime;
extern int SPARSE;
extern LR objFuncLR;
extern RR objFuncRR;
// Windows timer resolution
#if defined(_WIN32) || defined(_WIN64)
using Clock = std::chrono::high_resolution_clock;
#endif
// Linux/Unix timer resolution
#if defined(__linux__) || defined(__unix) || defined(__unix__)
// temporary
using Clock = std::chrono::high_resolution_clock;
// when extern variables placed here, endif seems not match to the previous
// unmatched one but closest one??
#endif
// void LogisticGradient(double *w, const mxArray *XtArray, double *y, double
// *G);

// void RidgeError(double *w, const mxArray *XtArray, double *y, double epoch,
// double telapsed, FILE *fp);

// void RidgeGradient(double *w, const mxArray *XtArray, double *y, double *G);

// void Shuffle(int *data, int num);

// double NoiseGen(double mean, double variance);

#endif
