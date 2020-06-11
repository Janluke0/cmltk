#!/usr/bin/env python3
import struct

def np_to_dat(array, f_name):
    with open(f_name,'wb') as f:
        if len(array.shape) == 2:
            a = struct.pack("=2Q"  ,*array.shape)
        else:
            a = struct.pack("=2Q", *(array.shape[0],1))
        a += struct.pack('=%sf' % array.size, *array.flatten())
        f.write(a)

def dat_to_np(f_name):
    with open(f_name, "rb") as f:
        pass


if __name__ == '__main__':
    import sklearn.datasets as ds 
    X, y = ds.load_breast_cancer(return_X_y = True)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
    np_to_dat(X_test,'breast_cancer.X.test.dat')
    np_to_dat(y_test,'breast_cancer.y.test.dat')
    np_to_dat(X_train,'breast_cancer.X.train.dat')
    np_to_dat(y_train,'breast_cancer.y.train.dat')
