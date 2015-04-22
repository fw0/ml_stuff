import statsmodels.nonparametric.kernel_regression as kernel_regression
import statsmodels.nonparametric.kernel_density as kernel_density
import python_utils.python_utils.sklearn_utils as sklearn_utils
import string

class kernel_regression_sklearn_wrapper(sklearn_utils.myTransformerMixin):

    def __init__(self, efficient=False, randomize=False, n_res=25, n_sub=50):
        self.EstimatorSettings = kernel_density.EstimatorSettings(efficient=efficient, randomize=randomize, n_res=n_res, n_sub=n_sub)
    
    def fit(self, X, y):
        self.predictor = kernel_regression.KernelReg(endog=y, exog=X, var_type = string.join(['c' for i in xrange(X.shape[1])],sep=''))
        return self

    def predict(self, X):
        return self.predictor.fit(X)[0]

    def get_params(self, deep=True):
        return {}
