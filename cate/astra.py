import numpy as np

from cate.xray import Geometry


class Detector:
    """Taken from my `reflex` package, since I don't want a direct dependency
    here."""

    def __init__(self, rows, cols, pixel_width, pixel_height):
        """
        :param rows: Number of horizontal rows
        :param cols: Number of vertical columns
        :param pixel_size: Total pixel size
        """
        self._rows = rows
        self._cols = cols
        self._pixel_width = pixel_width
        self._pixel_height = pixel_height

    @property
    def pixel_width(self):
        return self._pixel_width

    @property
    def pixel_height(self):
        return self._pixel_height

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def width(self):
        return self.pixel_width * self.cols

    @property
    def height(self):
        return self.pixel_height * self.rows

    def todict(self):
        return {'rows': self.rows,
                'cols': self.cols,
                'pixel_width': self.pixel_width,
                'pixel_height': self.pixel_height}


def crop_detector(detector: Detector, cols: int):
    if cols % 2 != 0:
        raise ValueError("`cols` must be dividable by 2.")

    return Detector(detector.rows, detector.cols - cols,
                    detector.pixel_width, detector.pixel_height)


def pixel2coord(pixel, det: Detector):
    """
    Annotated locations in the image frame do not directly correspond to good
    (x, y) coordinates in the detector convention. In our convention the
    detector midpoint is in (0, 0), the z-axis is pointing upwards and the
    image is flipped.
    """
    pixel[0] = -pixel[0] + det.cols  # revert vertical axis (image convention)
    pixel[1] = (pixel[1] - det.rows / 2) * det.pixel_width
    pixel[0] = (pixel[0] - det.cols / 2) * det.pixel_height
    pixel[1] = -pixel[1]  # observer frame is always flipped left-right

    return pixel


def pixels2coords(data, detector: Detector):
    for cam, proj_times in data.items():
        for proj in proj_times:
            for id, pixel in proj.items():
                pixel[:] = pixel2coord(pixel, detector)


def geom2astravec(g: Geometry, detector: dict):
    """CATE X-ray geom -> ASTRA vector description

    Note that the CATE description is dimensionless, so we use DET_PIXEL_WIDTH
    and DET_PIXEL_HEIGHT inside the function to go back to real dimensions."""
    c = lambda x: (x[1], x[0], - x[2])
    d = lambda x: np.array(
        (- x[1], - x[0], x[2])) * detector['pixel_width']
    e = lambda x: np.array(
        (- x[1], - x[0], x[2])) * detector['pixel_height']

    return [*c(g.source),
            *c(g.detector),
            *d(Geometry.u(g.roll, g.pitch, g.yaw)),
            *e(Geometry.v(g.roll, g.pitch, g.yaw))]
