#ifndef SIG_H
#define SIG_H
#include "comm.h"
#include "LogisticError.h"
#include "RidgeError.h"
using namespace Eigen;

void SIG_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, \
	 VectorXd &yTest, VectorXd &wtilde, VectorXd &G, string filename, double lambda=0.1, double eta=0.1, \
	int maxIter=60, int batchSize=1, int pass=20, double a=1, double b=1, double gamma=1,  int maxRunTime=20);

void SIG_logistic(VectorXd &w, const SparseMatrix<double> &Xt, int* innerIndices, int* outerStarts, const VectorXd &y, double lambda, double eta, VectorXd wtilde, VectorXd G, \
	int maxIter, int batchSize, int pass, int a, int b, int gamma, const MatrixXd &XtTest, \
	const VectorXd &yTest, int maxRunTime, string filename);


int SIG_LogisticInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &wtilde, VectorXd &G, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime);
int SIG_LogisticInnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd wtilde, double *G, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int SIG_LogisticInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &wtilde, VectorXd &G, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize, int maxRunTime);
int SIG_LogisticInnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd wtilde, double *G, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma);
int SIG_RidgeInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &wtilde, VectorXd &G, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime);
int SIG_RidgeInnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd wtilde, double *G, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int SIG_RidgeInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &wtilde, VectorXd &G, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize, int maxRunTime);
int SIG_RidgeInnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd wtilde, double *G, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma);


#endif 