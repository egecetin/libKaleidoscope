import pytest
import kaleidoscope


class TestSuite:
    def test_processing(self):
        nPixel = 1935 * 1088 * 3

        # Read data
        fileIn = open("tests/data/processing_1935x1088_InputData.bin", "rb")
        fileExpected = open("tests/data/processing_1935x1088_ExpectedData.bin", "rb")

        inImg = bytearray(fileIn.read())
        expectedImg = bytearray(fileExpected.read())
        outImg = bytearray(nPixel)

        # Process
        handle = kaleidoscope.PyKaleidoscope(6, 1935, 1088, 3, 0.45, 0.3)

        handle.processImage(inImg, outImg, nPixel)
        assert outImg == expectedImg

    def test_exception(self):
        k = 0.30
        scaleDown = 0.45
        n = 6
        width = 1935
        height = 1088
        nComponents = 3

        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(2, width, height, nComponents, scaleDown, k)

        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, 0, height, nComponents, scaleDown, k)
        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, -1, height, nComponents, scaleDown, k)

        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, 0, nComponents, scaleDown, k)
        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, -1, nComponents, scaleDown, k)

        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, height, 0, scaleDown, k)
        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, height, -1, scaleDown, k)

        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, height, nComponents, -0.01, k)
        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, height, nComponents, 1.01, k)
