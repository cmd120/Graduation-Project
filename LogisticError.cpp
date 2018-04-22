#include "include/LogisticError.h"

void LogisticError(const VectorXd &w, const MatrixXd &Xt, const VectorXd &y, double epoch, double telapsed, FILE *fp)
{
    int nSamples,nVars;
    long i;
    double tmp,sumError=0;
    nSamples = Xt.rows();
    nVars = Xt.cols();
    VectorXd tmpRes(nSamples);
    // sigmoid = 1./(1 + exp(-(X'*w)));
    // loglikelihood = mean(y.*log(sigmoid) + (1 - y).*log(1 - sigmoid));
    tmpRes = Xt.adjoint()*w;

    for(i = 0; i < nSamples; i++)
    {
        tmp = 1.0/(1 + exp(-tmpRes(i)));
        sumError += y(i) * log(tmp) + (1 - y(i)) * log(1 - tmp);
    }

    // print it, and save it
    // mexPrintf("%lf, %lf, %.25lf\n", epoch, telapsed, sumError*1.0/nSamples);
    // fprintf(fp, "%lf, %lf, %.25lf\n", epoch, telapsed, sumError*1.0/nSamples);
    // mxFree(tmpRes);
    return;
}