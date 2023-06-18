# distutils: language = c++

from libc.stdint cimport uint8_t
from kaleidoscope cimport Kaleidoscope

cdef class PyKaleidoscope:
    cdef Kaleidoscope *c_kaleidoscope  # Hold a C++ instance which we're wrapping

    def __init__(self, int nImage, int width, int height, int nComponents, double scaleDown, double dimConst):
        self.c_kaleidoscope = new Kaleidoscope(nImage, width, height, nComponents, scaleDown, dimConst)

    def processImage(self, uint8_t *inImg, uint8_t *outImg, size_t size):
        self.c_kaleidoscope.processImage(inImg, outImg, size)

    def __dealloc__(self):
        del self.c_kaleidoscope
