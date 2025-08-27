import pytest
import kaleidoscope


class TestSuite:
    def test_processing(self):
        n_pixel = 1935 * 1088 * 3

        # Read data
        file_in = open("tests/data/processing_1935x1088_InputData.bin", "rb")
        file_expected = open("tests/data/processing_1935x1088_ExpectedData.bin", "rb")

        in_img = bytearray(file_in.read())
        expected_img = bytearray(file_expected.read())
        out_img = bytearray(n_pixel)

        # Process
        handle = kaleidoscope.PyKaleidoscope(6, 1935, 1088, 3, 0.45, 0.3)

        handle.processImage(in_img, out_img, n_pixel)
        assert out_img == expected_img

    def test_exception(self):
        k = 0.30
        scale_down = 0.45
        n = 6
        width = 1935
        height = 1088
        n_components = 3

        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(2, width, height, n_components, scale_down, k)

        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, 0, height, n_components, scale_down, k)
        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, -1, height, n_components, scale_down, k)

        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, 0, n_components, scale_down, k)
        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, -1, n_components, scale_down, k)

        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, height, 0, scale_down, k)
        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, height, -1, scale_down, k)

        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, height, n_components, -0.01, k)
        with pytest.raises(Exception):
            kaleidoscope.PyKaleidoscope(n, width, height, n_components, 1.01, k)
