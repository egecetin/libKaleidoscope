from libc.stdint cimport uint8_t

cdef extern from "kaleidoscope.c":
    pass

cdef extern from "kaleidoscope.hpp" namespace "kalos":
    cdef cppclass Kaleidoscope:
        Kaleidoscope(int, int, int, int, double, double) except +
        void processImage(uint8_t *inImg, uint8_t *outImg, size_t)
