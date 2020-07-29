from clib import wrap_function, loadlib
import ctypes as ct
import numpy as np

m_element_t = ct.c_float
class m_Matrix(ct.Structure):
    _fields_ = [
        ("rows",ct.c_size_t),
        ("cols",ct.c_size_t),
        ("data",ct.POINTER(ct.c_float))
    ]
class Matrix:
    def __init__(self, arg):
        self._struct = None
        if isinstance(arg, Matrix):
            self._struct = arg._struct

        if isinstance(arg, m_Matrix):
            self._struct = arg            
        
        if isinstance(arg, np.ndarray):
            if len(arg.shape) == 2:
                r, c = arg.shape[0], arg.shape[1]
                m = m_empty(r, c)
                for i in range(0, r):
                    for j in range(0,c):
                        m.data[(i*c)+j] = arg[i,j]

            elif len(arg.shape) == 1:
                r = arg.shape[0]
                m = m_empty(r,1)
                for i in range(0,r):
                    m.data[i] = arg[i]
        
            self._struct = m
        if self._struct is None:
            raise AttributeError()

    @property
    def rows(self):
        return self._struct.rows

    @property
    def cols(self):
        return self._struct.cols

    @property
    def data(self):
        return self._struct.data

    def __repr__(self):
        return f"{self.__class__.__name__}(rows={self.rows}, cols={self.cols})"

    def __del__(self):
        m_free(ct.pointer(self._struct))

    def __array__(self):
        r, c, dat = self._struct.rows, self._struct.cols, self._struct.data
        return np.array([dat[i] for i in range(0,r*c)]).reshape(r,c)


    def __eq__(self, value):
        if not self is value:
            if self.rows != value.rows or self.cols != value.cols:
                return False
            for i in range(0,self.rows*self.cols):
                if not self.data[i] == value.data[i]:
                    return False
        return True

    def __ne__(self, value):
       return not self.__eq__(value)

    def __neg__(self):
        return Matrix(m_mul_scalar(self._struct,-1.))

    def __matmul__(self, other): 
        if isinstance(other,self.__class__):
            if self.rows != other.rows:
                raise ValueError("the matrices must have the same number of rows")
            return Matrix(m_dot_T(self._struct,other._struct))
        raise TypeError()

    def __add__(self, other):
        if isinstance(other,self.__class__):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("the matrix must have the same sizes")
            return  Matrix(m_sum(self._struct,other._struct))
        if isinstance(other,float) or  isinstance(other,int):
            return  Matrix(m_sum_scalar(self._struct,other))

        raise TypeError()

    def __sub__(self, other):
        if isinstance(other,self.__class__):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("the matrices must have the same sizes")
            return  Matrix(m_sub(self._struct,other._struct))
        if isinstance(other,float) or  isinstance(other,int):
            return  Matrix(m_sum_scalar(self._struct,-other))
        raise TypeError()

    def __mul__(self, other):
        if isinstance(other,self.__class__):
            if self.cols != other.rows:
                raise ValueError("the second matrix must have a rows number equals to the first cols number")
            return Matrix(m_dot(self._struct,other._struct))
        if isinstance(other,float) or  isinstance(other,int):
            return Matrix(m_mul_scalar(self._struct,other))
        raise TypeError()

    def __truediv__(self, other):
        if isinstance(other,float) or  isinstance(other,int):
            return Matrix(m_mul_scalar(self._struct,1/other))
        raise TypeError()

    def save(self, path):        
        return m_store(self._struct, path.encode('UTF-8')) == 0
    
    @property
    def T(self):
        return Matrix(m_transpose(self._struct))

    @staticmethod
    def load(path):
        return Matrix(m_load(path.encode('UTF-8')))


        

lib = loadlib("matrix")



def wrap_m_op(funcname,args):
    return wrap_function(lib,f"M_{funcname}", m_Matrix.from_address, args) 

m_empty = wrap_m_op("new", [ct.c_size_t, ct.c_size_t]) 
m_zeros = wrap_m_op("zeros",  [ct.c_size_t, ct.c_size_t]) 
m_ones = wrap_m_op("ones",  [ct.c_size_t, ct.c_size_t]) 
m_identity = wrap_m_op("identity",  [ct.c_size_t]) 

m_free = wrap_function(lib,"M_free", ct.c_void_p, [ct.c_void_p]) 


m_rand = wrap_m_op("rand",  [ct.c_size_t, ct.c_size_t, m_element_t, m_element_t]) 
m_rand_seed = wrap_function(lib,"srand", ct.c_void_p, [ct.c_uint]) 

m_load = wrap_m_op("load", [ct.c_char_p]) 
m_copy = wrap_m_op("copy", [ct.c_char_p]) 
m_store = wrap_function(lib,"M_store", ct.c_int, [ct.POINTER(m_Matrix), ct.c_char_p]) 

m_print = wrap_function(lib,"M_print", None, [ct.POINTER(m_Matrix)]) 

m_dot = wrap_m_op("dot",[ct.POINTER(m_Matrix), ct.POINTER(m_Matrix)])
m_dot_T = wrap_m_op("dot_T",[ct.POINTER(m_Matrix), ct.POINTER(m_Matrix)])
m_sum = wrap_m_op("sum",[ct.POINTER(m_Matrix), ct.POINTER(m_Matrix)])
m_sub = wrap_m_op("sub",[ct.POINTER(m_Matrix), ct.POINTER(m_Matrix)])

m_transpose = wrap_m_op("transpose",[ct.POINTER(m_Matrix)])
#m_invert = wrap_m_op("invert",[ct.POINTER(m_Matrix)])

m_mul_scalar = wrap_m_op("mul_scalar",[ct.POINTER(m_Matrix), m_element_t])
m_sum_scalar = wrap_m_op("sum_scalar",[ct.POINTER(m_Matrix), m_element_t])
    
def ones(rows,cols=None):
    if cols is None:
        cols = rows
    return Matrix(m_ones(rows,cols))

def identity(rows,cols=None):
    if cols is None:
        cols = rows
    return Matrix(m_identity(rows,cols))

def zeros(rows, cols=None):
    if cols is None:
        cols = rows
    return Matrix(m_zeros(rows,cols))

def rand(rows,cols=None,min=-1,max=1, seed=42):
    if cols is None:
        cols = rows
    m_rand_seed(seed)
    return Matrix(m_rand(rows,cols,min,max))

if __name__ == "__main__":
    #import pathlib,os
    #p = str(pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent / "datasets" / "breast_cancer.X.test.dat")
    #m = m_load(p.encode('UTF-8'))
    #m_print(m)
    
    m = rand(40,40, seed=-31)
    m.save("file.dat")
    #m_print(m)
    #m_print(m)

    #m_print(m)
    print()
    m = -m
    m += 1
    m += m
    m -= 1
    m -= m*0.5
    m *= m
    m = m/ 2
    #m_print(m)
    print()
    m = m.T
    #m_print(m)
    print(m==m)
    print(ones(10) is ones(10))
    print(ones(10) == ones(10))
    print(ones(10) == ones(10,1))
    print(ones(10) == zeros(10))
    
    m = identity(19)
    m += m
    a = np.asarray(m)
    a += 1
    print(a)
    print(a.mean(axis=1))