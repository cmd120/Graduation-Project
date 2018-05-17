#include "include/LogisticGradient.h"

void LogisticGradient(Eigen::VectorXd &wtilde, Eigen::VectorXd &G,
                      const Eigen::MatrixXd &Xt, Eigen::VectorXd &y) {
  long i, j;
  int nVars, nSamples;
  nVars = Xt.rows();
  nSamples = Xt.cols();
  Eigen::VectorXd tmpRes(nSamples);
  // clear G
  G = Eigen::MatrixXd::Zero(nVars, 1);
  tmpRes = Xt.adjoint() * wtilde;
  for (i = 0; i < nVars; ++i) {
    tmpRes(i) = (1.0 / (1 + exp(-tmpRes(i))) - y(i)) / nSamples;
  }
  G = Xt * tmpRes;
  return;
}
void LogisticGradient(Eigen::VectorXd &wtilde, Eigen::VectorXd &G,
                      int *innerIndices, int *outerStarts,
                      const Eigen::SparseMatrix<double> &Xt,
                      Eigen::VectorXd &y) {
  long i, j;
  int nVars, nSamples;
  double innerProd;
  nVars = Xt.rows();
  nSamples = Xt.cols();
  Eigen::VectorXd tmpRes(nSamples);
  // clear G
  G = Eigen::MatrixXd::Zero(nVars, 1);
  for (i = 0; i < nSamples; ++i) {
    innerProd = 0;
    j = outerStarts[i];
    for (Eigen::SparseMatrix<double>::InnerIterator it(Xt, i); it; ++it, ++j) {
      innerProd += wtilde[innerIndices[j]] * it.value();
    }
    tmpRes[i] = innerProd;
  }
  for (i = 0; i < nVars; ++i) {
    tmpRes(i) = (1.0 / (1 + exp(-tmpRes(i))) - y(i)) / nSamples;
  }
  for (i = 0; i < nSamples; ++i) {
    innerProd = 0;
    j = outerStarts[i];
    for (Eigen::SparseMatrix<double>::InnerIterator it(Xt, i); it; ++it, ++j) {
      G[innerIndices[j]] += tmpRes[i] * it.value();
    }
  }
  return;
}