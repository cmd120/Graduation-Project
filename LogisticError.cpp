#include "include/LogisticError.h"

ERRORCODE LogisticError(const VectorXd &w, const MatrixXd &Xt,
                        const VectorXd &y, double epoch, double telapsed,
                        FILE *fp) {
  ERRORCODE ret = DEFAULT;
  int nSamples, nVars;
  long i;
  double tmp, sumError = 0;
  nSamples = Xt.cols();
  nVars = Xt.rows();
  VectorXd tmpRes(nSamples);
  cout << "sum of w:" << w.sum() << endl;
  tmpRes = Xt.adjoint() * w;
  for (i = 0; i < nSamples; i++) {
    double debugInfo;
    tmp = 1.0 / (1 + exp(-tmpRes(i)));
    if (tmp == 1) {
      if (y(i) == 1)
        continue;
      else
        cout << "Problem tmpRes(i): " << tmpRes(i) << endl;
    }
    sumError += y(i) * log(tmp) + (1 - y(i)) * log(1 - tmp);
  }
  fprintf(fp, "%lf, %lf, %.25lf\n", epoch, telapsed, sumError * 1.0 / nSamples);
  return ret = SUCCESS;
}
ERRORCODE LogisticError(const VectorXd &w, const SparseMatrix<double> &Xt,
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
  cout << "tmpRes sum: " << tmpRes.sum() << endl;
  for (i = 0; i < nSamples; i++) {
    double debugInfo;
    tmp = 1.0 / (1 + exp(-tmpRes(i)));
    if (tmp == 1) {
      if (y(i) == 1)
        continue;
      else
        cout << "Problem tmpRes(i): " << tmpRes(i) << endl;
    }
    sumError += y(i) * log(tmp) + (1 - y(i)) * log(1 - tmp);
  }
  fprintf(fp, "%lf, %lf, %.25lf\n", epoch, telapsed, sumError * 1.0 / nSamples);
  return ret = SUCCESS;
}