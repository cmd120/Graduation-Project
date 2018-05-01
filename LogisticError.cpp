#include "include/LogisticError.h"

void LogisticError(const VectorXd &w, const MatrixXd &Xt, const VectorXd &y, double epoch, double telapsed, FILE *fp)
{
    // cout << "enter LogisticError" << endl;
    int nSamples,nVars;
    long i;
    double tmp,sumError=0;
    nSamples = Xt.cols();
    nVars = Xt.rows();
    VectorXd tmpRes(nSamples);
    // sigmoid = 1./(1 + exp(-(X'*w)));
    // loglikelihood = mean(y.*log(sigmoid) + (1 - y).*log(1 - sigmoid));
    cout << "sum of w:" << w.sum() << endl;
    tmpRes = Xt.adjoint()*w;

    // cout << "tmpRes pass" << endl;
    // cout << "y size: " << y.size() << endl;
    // cout << "tmpRes size" << tmpRes.size() <<endl;
    // cout << "LogisticError point1" << endl;
    // cout << y.size() << endl;
    // cout << tmpRes.size() << endl;
    // cout << nSamples << endl;

    // cout << tmpRes << endl;
    for(i = 0; i < nSamples; i++)
    {
        double debugInfo;
        tmp = 1.0/(1 + exp(-tmpRes(i)));
        if(tmp<=0 || tmp>=1){
            cout << "tmp: " << tmp << endl;
            cout << "problem tmpRes(i): " << tmpRes(i) << endl;
            return; 
        }
        // cout << "tmp" << tmp << endl;
        debugInfo =  y(i) * log(tmp) + (1 - y(i)) * log(1 - tmp);
        // cout << debugInfo << endl;
        sumError += y(i) * log(tmp) + (1 - y(i)) * log(1 - tmp);
    }
    // cout << "LogisticError point2" << endl;
    // print it, and save it
    // mexPrintf("%lf, %lf, %.25lf\n", epoch, telapsed, sumError*1.0/nSamples);
    // cout << "sumError" << sumError <<endl;
    fprintf(fp, "%lf, %lf, %.25lf\n", epoch, telapsed, sumError*1.0/nSamples);
    // mxFree(tmpRes);
    // cout << "leave LogisticError" << endl;
    return;
}