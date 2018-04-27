#ifndef IAG_H
#define IAG_H
#include "comm.h"
#include "LogisticError.h"
#include "RidgeError.h"
using namespace Eigen;
using namespace std;

void IAG_logistic(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, \
	VectorXd &yTest, VectorXd &d, VectorXd &g, string filename, double lambda=0.035, double eta=0.1, \
	int batchSize=1, int pass=10, double a=1e-2, double b=0, double gamma=0,  int maxRunTime=60);

void IAG_logistic(VectorXd &w, const SparseMatrix<double> &Xt, int* innerIndices, int* outerStarts, const VectorXd &y, double lambda, double eta, VectorXd d, VectorXd g, \
	int maxIter, int batchSize, int pass, int a, int b, int gamma, const MatrixXd &XtTest, \
	const VectorXd &yTest, int maxRunTime, string filename);


int IAG_LogisticInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime);
int IAG_LogisticInnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int IAG_LogisticInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize, int maxRunTime);
int IAG_LogisticInnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma);
int IAG_RidgeInnerLoopSingleDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int maxRunTime);
int IAG_RidgeInnerLoopSingleSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize, int pass, double a, double b, double gamma);
int IAG_RidgeInnerLoopBatchDense(VectorXd &w, const MatrixXd &Xt, VectorXd &y, const MatrixXd &XtTest, VectorXd &yTest, VectorXd &d, VectorXd &g, double lambda, long maxIter, int nSamples, int nVars, int pass, double a, double b, double gamma, int batchSize, int maxRunTime);
int IAG_RidgeInnerLoopBatchSparse(VectorXd &w, const SparseMatrix<double> &Xt, int *innerIndices, int *outerStarts, VectorXd y, double lambda, VectorXd d, double *g, long maxIter, int nSamples, int nVars, int batchSize,int pass, double a, double b, double gamma);




#endif 