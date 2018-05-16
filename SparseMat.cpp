#include "include/SparseMat.h"

// create outerStarts array(), jc is counterpart in matlab
void InitOuterStarts(const SparseMatrix<double> &mat, int *outerStarts) {
  *outerStarts = 0;
  int count = 0;
  for (int k = 0; k < mat.outerSize(); ++k) {
    SparseMatrix<double>::InnerIterator it(mat, k);
    int nextColFirstIndex = 0;
    if (it) {
      while (it) {
        nextColFirstIndex++;
        ++it;
      }
      outerStarts[count + 1] = outerStarts[count] + nextColFirstIndex;
      count++;
    }
  }
  // DEBUG
  // for(int i=0;i<count+1;++i){
  // 	cout << outerStarts[i] <<endl;
  // }
  return;
}

int issparse(vector<double> &mat) {
  int ret;
  long count = 0;
  // cout << "mat size: " << mat.size() << endl;
  for (long i = 0; i < mat.size(); ++i) {
    count = (mat[i] - 0) < ACCURACY ? count : count + 1;
  }
  ret = count >= mat.size() / 2 ? 0 : 1;
  return ret;
}

void algorithmInit(SparseMatrix<double> &Xt, VectorXd &w,
                   SparseMatrix<double> &XtTest, VectorXd &yTest,
                   double &lambda, double &eta, double &a, double &b,
                   double &gamma, int &maxIter, int &batchSize, int &passes,
                   int &maxRunTime, string &filename, int &datasetNum) {
  cout << "Input batchSize: " << endl;
  cin >> batchSize;
  filename = filename + "_output_sparse_" + to_string(batchSize);
  fp = fopen(filename.c_str(), "w");
  if (fp == NULL) {
    cout << "Cannot write results to file: " << filename << endl;
  }
  LogisticError(w, XtTest, yTest, 0, 0, fp);
  epochCounter = (epochCounter + 1) % PRINT_FREQ;
  VectorXd tmp = Xt.col(0);
  double L = tmp.array().square().sum() / 4 + lambda;
  lambda = 1 / Xt.cols();
  eta = 0.1;
  // DEBUG
  a = 1e-8;
  b = 0;
  maxIter = 2 * Xt.cols();
  passes = 6e2;
  maxRunTime = 100;
  double mu = lambda;
  objFuncLR = LR(lambda, L, mu);
  if (DEBUG) {
    cout << "enter step length:" << endl;
    cin >> a;
    cout << "enter passes:" << endl;
    cin >> passes;
  }
  startTime = Clock::now();
  return;
}
