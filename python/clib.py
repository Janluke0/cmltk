import ctypes as ct
import os, pathlib


def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

def loadlib(name):
    libname = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))  / ".." / "bin" / f"{name}.so"
    c_lib = ct.CDLL(libname)
    return c_lib
