#include "include/RidgeError.h"

ERRORCODE RidgeError(const VectorXd &w, const MatrixXd &Xt, const VectorXd &y,
                     double epoch, double telapsed, FILE *fp) {
  ERRORCODE ret = DEFAULT;
  int nSamples, nVars;
  long i;
  double tmp, sumError = 0;
  nSamples = Xt.cols();
  nVars = Xt.rows();
  VectorXd tmpRes(nSamples);
  tmpRes = Xt.adjoint() * w;
  for (i = 0; i < nSamples; i++) {
    double tmp = tmpRes(i) - y(i);
    sumError += tmp * tmp;
  }
  fprintf(fp, "%lf, %lf, %.25lf\n", epoch, telapsed, sumError * 1.0 / nSamples);
  return ret = SUCCESS;
}
ERRORCODE RidgeError(const VectorXd &w, const SparseMatrix<double> &Xt,
                     const VectorXd &y, double epoch, double telapsed,
                     FILE *fp) {
  ERRORCODE ret = DEFAULT;
  int nSamples, nVars;
  long i;
  double tmp, sumError = 0;
  nSamples = Xt.cols();
  nVars = Xt.rows();
  VectorXd tmpRes(nSamples);
  tmpRes = Xt.adjoint() * w;
  for (i = 0; i < nSamples; i++) {
    double tmp = tmpRes(i) - y(i);
    sumError += tmp * tmp;
  }
  fprintf(fp, "%lf, %lf, %.25lf\n", epoch, telapsed, sumError * 1.0 / nSamples);
  return ret = SUCCESS;
}