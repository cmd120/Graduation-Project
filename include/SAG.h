#ifndef SAG_H
#define SAG_H
#include "comm.h"
#include "LogisticError.h"
#include "RidgeError.h"
using namespace Eigen;

void SAG_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, \
	 VectorXd &yTest, VectorXd &d, VectorXd &g, string filename, double lambda=0.1, double eta=0.1, \
	int maxIter=60, int batchSize=1, int pass=20, double a=1, double b=1, double gamma=1,  int maxRunTime=60);

void SAG_logistic(VectorXd &w, const SparseMatrix<double> &Xt, int* innerIndices, int* outerStarts, const VectorXd &y, double lambda, double eta, VectorXd d, VectorXd g, \
	int maxIter, int batchSize, int pass, double a, double b, int gamma, const MatrixXd &XtTest, \
	 VectorXd &yTest, int maxRunTime, string filename);


int SAG_LogisticInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd y, const MatrixXd &XtTest, VectorXd yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime);
int SAG_LogisticInnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int SAG_LogisticInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd y, const MatrixXd &XtTest, VectorXd yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize, int maxRunTime);
int SAG_LogisticInnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma);
int SAG_RidgeInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime);
int SAG_RidgeInnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int SAG_RidgeInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize, int maxRunTime);
int SAG_RidgeInnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma);





#endif 