from clib import loadlib, wrap_function
from matrix import Matrix, m_element_t, m_Matrix
import ctypes as ct

lib = loadlib("logreg")
def wrap_logreg_op(funcname, ret, args):
    return wrap_function(lib,f"LOGREG_{funcname}", ret, args) 

class CLogisticRegression(ct.Structure):
    _fields_ = [
        ("m", ct.c_int),
        ("weights", m_Matrix),
        ("bias", m_element_t)
    ]


class LogisticRegression():
    def __init__(self):
        self._struct = None
    
    def fit(self,X,y,_lambda=0.1,max_it=100,tol=1e-4):
        if self._struct is None:
            self._struct = lr_new(X.cols)
        return lr_train(self._struct, _lambda, X._struct, y._struct, max_it, tol)

    def predict(self,X):
        if self._struct is None:
            raise AttributeError("Model not trained yet")
        return Matrix(lr_inference(self._struct,X._struct))



lr_new = wrap_logreg_op("new", CLogisticRegression.from_address, [ct.c_size_t])

#iteration
#lr_iteration = wrap_logreg_op("iteration", LogisticRegression.from_address, [ct.c_size_t])
#train
"""int     LOGREG_train    (LOGREG_model_t *model, // the result of training
                        float lambda,   //lambda factor for L2 regularization
                        matrix_t *X,    //observation matrix (m,n) 
                        matrix_t *y,    //labels vector (m,1)
                        int max_it, 
                        double loss_tol);"""
# model, lambda, X, y, max_it, loss_tol
lr_train = wrap_logreg_op("train", ct.c_void_p, 
            [ct.POINTER(CLogisticRegression), ct.c_float, ct.POINTER(m_Matrix), ct.POINTER(m_Matrix), ct.c_int, ct.c_double])


#inference
lr_inference = wrap_logreg_op("inference", m_Matrix.from_address, 
            [ct.POINTER(CLogisticRegression), ct.POINTER(m_Matrix)])

if __name__ == "__main__":
    print(lr_new(10).weights)