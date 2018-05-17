#include "include/LogisticError.h"

ERRORCODE LogisticError(const Eigen::VectorXd &w, const Eigen::MatrixXd &Xt,
                        const Eigen::VectorXd &y, double epoch, double telapsed,
                        FILE *fp) {
  ERRORCODE ret = DEFAULT;
  int nSamples, nVars;
  long i;
  double tmp, sumError = 0;
  nSamples = Xt.cols();
  nVars = Xt.rows();
  Eigen::VectorXd tmpRes(nSamples);
  std::cout << "sum of w:" << w.sum() << std::endl;
  tmpRes = Xt.adjoint() * w;
  for (i = 0; i < nSamples; i++) {
    double debugInfo;
    tmp = 1.0 / (1 + exp(-tmpRes(i)));
    if (tmp == 1) {
      if (y(i) == 1)
        continue;
      else
        std::cout << "Problem tmpRes(i): " << tmpRes(i) << std::endl;
    }
    sumError += y(i) * log(tmp) + (1 - y(i)) * log(1 - tmp);
  }
  fprintf(fp, "%lf, %lf, %.25lf\n", epoch, telapsed, sumError * 1.0 / nSamples);
  return ret = SUCCESS;
}
ERRORCODE LogisticError(const Eigen::VectorXd &w,
                        const Eigen::SparseMatrix<double> &Xt,
                        const Eigen::VectorXd &y, double epoch, double telapsed,
                        FILE *fp) {
  ERRORCODE ret = DEFAULT;
  int nSamples, nVars;
  long i;
  double tmp, sumError = 0;
  nSamples = Xt.cols();
  nVars = Xt.rows();
  Eigen::VectorXd tmpRes(nSamples);
  tmpRes = Xt.adjoint() * w;
  std::cout << "tmpRes sum: " << tmpRes.sum() << std::endl;
  for (i = 0; i < nSamples; i++) {
    double debugInfo;
    tmp = 1.0 / (1 + exp(-tmpRes(i)));
    if (tmp == 1) {
      if (y(i) == 1)
        continue;
      else
        std::cout << "Problem tmpRes(i): " << tmpRes(i) << std::endl;
    }
    sumError += y(i) * log(tmp) + (1 - y(i)) * log(1 - tmp);
  }
  fprintf(fp, "%lf, %lf, %.25lf\n", epoch, telapsed, sumError * 1.0 / nSamples);
  return ret = SUCCESS;
}