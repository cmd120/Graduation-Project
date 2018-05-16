#include "include/DenseMat.h"

VectorXd LogisticPartialGradient(VectorXd &innerProd, VectorXd &y) {
  return (1 + (-innerProd).array().exp()).inverse() - y.array();
}
double LogisticPartialGradient(double innerProd, double y) {
  return 1 / (1 + exp(-innerProd)) - y;
}

VectorXd RidgePartialGradient(VectorXd &innerProd, VectorXd &y) {
  return innerProd - y;
}
double RidgePartialGradient(double innerProd, double y) {
  return innerProd - y;
}

void algorithmInit(MatrixXd &Xt, VectorXd &w, MatrixXd &XtTest, VectorXd &yTest,
                   double &lambda, double &eta, double &a, double &b,
                   double &gamma, int &maxIter, int &batchSize, int &passes,
                   int &maxRunTime, string &filename, int &datasetNum) {
  cout << "Input batchSize: " << endl;
  cin >> batchSize;
  filename = filename + "_output_dense_" + to_string(batchSize);
  fp = fopen(filename.c_str(), "w");
  if (fp == NULL) {
    cout << "Cannot write results to file: " << filename << endl;
  }
  LogisticError(w, XtTest, yTest, 0, 0, fp);
  epochCounter = (epochCounter + 1) % PRINT_FREQ;
  switch (datasetNum) {
    case 1:
      double L = Xt.col(0).array().square().sum() / 4 + lambda;
      double mu;
      lambda = 1 / Xt.cols();
      eta = 0.1;
      a = batchSize >= 2 ? 1 : 1e-2;
      b = 0;
      gamma = 0;
      maxIter = 2 * Xt.cols();
      passes = 10;
      maxRunTime = 60;
      mu = lambda;
      objFuncLR = LR(lambda, L, mu);
      break;
  }
  if (DEBUG) {
    cout << "enter step length:" << endl;
    cin >> a;
    cout << "enter passes:" << endl;
    cin >> passes;
  }
  startTime = Clock::now();
  return;
}