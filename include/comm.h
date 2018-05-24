// comm.h declares basic libraries, global variables and classes across files
#ifndef COMM_H
#define COMM_H

#define DEBUG 1        ///< 0 for release, 1 for debug
#define NOISY 0        ///< 0 for optimization, 1 for bayes model
#define PRINT_FREQ 50  ///< print test error PRINT_FREQ times per epoch
#define BILLION 1E9
#define FILE_NAME_LENGTH 64  ///< the maximum length of file name
#define ACCURACY 10E-5
//#define EIGEN_USE_MKL_ALL ///< if Intel MKL is installed, uncomment this line

/// An global enum type for function return type.
/// Use ERRORCODE to know if function works properly and to handle exception
///
typedef enum {
  DEFAULT = 0,         ///< unassigned
  SUCCESS,             ///< function return successfully
  STEPLENGTHTOOBIG,    ///< the given step size is too big
  STEPLENGTHTOOSMALL,  ///< the given step size is too small
  ALLOCERROR,          ///< run out of space
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
/// Logistic Regression class.
/// This class consists of functions compute the cost, likelihood, hypotheis and
/// etc. under logistic regression problem.
///
class LR {
 private:
  double lambda = 0;
  double L = 0;
  double mu = 0;
  void sigfunc(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y,
               Eigen::VectorXd &sigmoid) {
    sigmoid = (1 + (-X.adjoint() * w).array().exp()).cwiseInverse();
  }

 public:
  Eigen::VectorXd optSolution;  ///< the optimal w
  double optCost;               ///< the optimal cost
  ///
  /// A construtor.
  ///
  LR(double lambda = 0, double L = 0, double mu = 0) {
    this->lambda = lambda;
    this->L = L;
    this->mu = mu;
  }
  ///
  /// Computes the total cost.
  ///
  double costfunc(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y) {
    Eigen::VectorXd sigmoid;
    sigfunc(w, X, y, sigmoid);
    Eigen::VectorXd siglog = sigmoid.array().log();
    Eigen::VectorXd siglogimpl = (1 - sigmoid.array()).log();
    double loss =
        -(y.dot(siglog) + (1 - y.array()) * siglogimpl.array()).mean();
    double regularizer = this->lambda / 2 * (w.array().square()).sum();
    return loss + regularizer;
  }
  ///
  /// Computes log-likelihood.
  ///
  double loglikelihood(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                       Eigen::VectorXd &y) {
    Eigen::VectorXd sigmoid;
    sigfunc(w, X, y, sigmoid);
    double loglikelihood = (y.array() * sigmoid.array().log() +
                            (1 - y.array()) * (1 - sigmoid.array()).log())
                               .mean();
    return loglikelihood;
  }
  ///
  /// Computes gradient without reularizer.
  ///
  Eigen::VectorXd grad_noreg(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                             Eigen::VectorXd &y) {
    int d = X.rows(), n = X.cols();
    Eigen::VectorXd tmpExp = (-(X.adjoint() * w)).array().exp();
    Eigen::VectorXd gradient =
        X * (1 + tmpExp.array() - y.array()).matrix().cwiseInverse() / n;
    return gradient;
  }
  ///
  /// Computes hypothesis.
  ///
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
  ///
  /// Computes the prediction.
  ///
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
  ///
  /// Computes the score of the current w.
  ///
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
  ///
  /// A destrutor.
  ///
  ~LR() { ; };
};
/// Ridge Regression class.
/// This class consists of functions compute the cost, likelihood, hypotheis and
/// etc. under ridge regression problem.
///
class RR {
 private:
  double lambda = 0;
  double L = 0;
  double mu = 0;

 public:
  Eigen::VectorXd optSolution;  ///< the optimal w
  double optCost;               ///< the optimal cost
  ///
  /// A construtor.
  ///
  RR(double lambda = 0, double L = 0, double mu = 0) {
    this->lambda = lambda;
    this->L = L;
    this->mu = mu;
  }
  ///
  /// Computes the total cost.
  ///
  double costfunc(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y) {
    Eigen::VectorXd sigmoid;
    double loss = (y - X.adjoint() * w).array().square().mean() / 2;
    double regularizer = this->lambda / 2 * (w.array().square()).sum();
    return loss + regularizer;
  }
  ///
  /// Computes log-likelihood.
  ///
  double loglikelihood(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                       Eigen::VectorXd &y) {
    Eigen::VectorXd innerProd;
    innerProd = X.adjoint() * w;
    double loglikelihood = 1 / 2 * (innerProd - y).array().square().mean();
    return loglikelihood;
  }
  ///
  /// Computes gradient without reularizer.
  ///
  Eigen::VectorXd grad_noreg(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                             Eigen::VectorXd &y) {
    int d = X.rows(), n = X.cols();
    Eigen::VectorXd gradient = 1 / n * (X * (X.adjoint() * w - y));
  }
  ///
  /// Computes hypothesis.
  ///
  void hypothesis(Eigen::VectorXd &w, Eigen::MatrixXd &X,
                  Eigen::VectorXd &hypothesis) {
    hypothesis = X.adjoint() * w;
  }
  ///
  /// Prints the cost.
  ///
  // void costprint(Eigen::VectorXd &w, Eigen::MatrixXd &X, Eigen::VectorXd &y,
  //                int stage) {
  //   double cost = costfunc(w, X, y);
  //   double grad_square = (grad_noreg(w, X, y) + this->lambda * w).norm();
  //   // fprintf('epoch: %4d, cost: %.25f, grad: %.25f\n', stage, cost,
  //   // grad_square);
  //   // fprintf('epoch: %4d, cost: %.25f\n', stage, cost);  // resize the cost
  // }

  ///
  /// Computes the prediction.
  ///
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
  ///
  /// Computes the score of the current w.
  ///
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
  ///
  /// A destrutor.
  ///
  ~RR() { ; };
};
/// Algorithm class.
/// Object of this class contains necessary information an algorithm need to
/// work, such as step size, batch size.
///
class ALGORITHM {
 private:
  std::string name;  ///< algorithm name
  double step;       ///< step size
  double batchSize;  ///< batch size

 public:
  ///
  /// A constructor.
  ///
  ALGORITHM(std::string name, double step, double batchSize)
      : name(name), step(step), batchSize(batchSize) {
    ;
  }
  ///
  /// Get the name of algorithm.
  ///
  std::string getname() { return this->name; }
  ///
  /// Change the step size.
  ///
  int setsize(double step) {
    this->step = step;
    return 0;
  }
};
///
/// Class represents a dataset for later training or validation.
///
class DATASET {
 private:
  std::string name;           ///< dataset name
  int features;               ///< # of features.
  std::string testfilename;   ///< test file name
  std::string trainfilename;  ///< train file name
  int trainsize;              ///< # of train samples
  int testsize;               ///< # of test samples

 public:
  ///
  /// Default constructor(do nothing inside)
  ///
  DATASET() { ; }
  ///
  /// A constructor.
  /// Use this constructor to initialize a valid dataset object.
  ///
  DATASET(std::string name, int features, std::string testfilename,
          std::string trainfilename, int trainsize, int testsize)
      : name(name),
        features(features),
        trainfilename(trainfilename),
        testfilename(testfilename),
        trainsize(trainsize),
        testsize(testsize) {
    ;
  }
  ///
  /// Return the # of features
  ///
  int getfeatures() { return this->features; }
  ///
  /// Return the # of training samples
  ///
  int gettrainsize() { return this->trainsize; }
  ///
  /// Return the # of testing samples
  ///
  int gettestsize() { return this->testsize; }
  ///
  /// Return the name of trainning file
  ///
  std::string gettrainfilename() { return this->trainfilename; }
  ///
  /// Return the name of testing file
  ///
  std::string gettestfilename() { return this->testfilename; }
};
extern int epochCounter;  ///< count the current epoch time (global)
extern FILE *fp;          ///< the file pointer (global)
extern std::chrono::high_resolution_clock::time_point
    startTime;        ///< time base (global)
extern int SPARSE;    ///< 1 if current dataset is sparse, else 0 (global)
extern LR objFuncLR;  ///< object for Logistic Regression (global)
extern RR objFuncRR;  ///< object for Ridge Regression (global)
/// Windows timer resolution
#if defined(_WIN32) || defined(_WIN64)
using Clock = std::chrono::high_resolution_clock;
#endif
/// Linux/Unix timer resolution
#if defined(__linux__) || defined(__unix) || defined(__unix__)
/// temporary
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
