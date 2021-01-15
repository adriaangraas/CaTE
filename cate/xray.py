from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import transforms3d

from cate.param import (Parameter, VectorParameter,
                        params2ndarray, update_params)


class MarkerLocation(VectorParameter):
    pass


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


class StaticGeometry:
    """Geometry description for a single angle without moving parts.

    I see two sensible ways to describe the detector vectors in a geometry.
        1.  By setting the detector frame to be orthogonal to the source-
            detector line.
        2.  By setting the detector frame statically, for one convention.

    The problem with (1) is that a detector tilt has to be described differently
    when the source/detector positions move w.r.t. to each other. The problem
    with (2) is that you'd need to describe/calculate the tilt in a way that
    is not the "deviation from the wanted detector". The problem really
    falls outside of this class and depends on the context of the dynamic
    acquisition. Therefore we take the most simple/primitive description which
    is (2).

    In practice, this means that the user has to specify the detector u, v,
    and when they are left empty, they will be orthogonal to the source-detector
    line. Internally the 6-param (u,v) will be parameterized to (r,p,y) so that
    there is no information redundancy. The user can also specify (r,p,y)
    directly.

    TODO: I'll probably have to make some changes to differentiate between
        the detector location and the (u,v) starting points when the ROI of
        a detector is not centered around the middle of the detector. This
        would cause a misplaced origin in the detector image.
    """
    ANGLES_CONVENTION = "sxyz"

    def __init__(
        self,
        source: Any,
        detector: Any,
        roll: Any = None,
        pitch: Any = None,
        yaw: Any = None,
        u: np.ndarray = None,
        v: np.ndarray = None,
    ):
        """Initialialize geometry from source and detector

        The detector frame are the three axis in the detector point where
        the vertical axis is always perfectly upward [0, 0, 1], but the
        in the detector point.
        The u and v that span the detector plane are chosen to be

        """
        self._source = source
        self._detector = detector
        self._roll = roll
        self._pitch = pitch
        self._yaw = yaw

        n = self.source - self.detector  # in global frame coordinates
        n /= np.linalg.norm(n)

        if self.roll is None or self.pitch is None or self.yaw is None:
            if u is None or v is None:
                # find v as the z-det vector orthogonal to the detector-to-source vec
                # (Gram-Schmidt orthogonalizion)
                z = np.array([0, 0, 1.])
                v = z - n.dot(z) / np.dot(n, n) * n
                len_v = np.linalg.norm(v)

                if len_v == 0.:
                    raise NotImplementedError(
                        "Geometries with perfectly vertical source-"
                        "detector line are not supported, because I didn't want"
                        " to write unpredictable logic. Upright your geom or"
                        " check if axes correspond to [x, y, z].")

                v /= len_v

                # find u as the vector orthogonal to `v` and the `detector-to-source`
                # u = np.cross(vec, v)  # right-hand-rule = right direction?
                u = np.cross(z, n)
                u /= np.linalg.norm(u)

            np.testing.assert_almost_equal(np.dot(u, v), 0.)
            np.testing.assert_almost_equal(np.dot(u, n), 0.)
            np.testing.assert_almost_equal(np.dot(v, n), 0.)

            mat = np.array([n, u, v])
            self.roll, self.pitch, self.yaw = self.mat2angles(mat)
        else:
            if roll is None or pitch is None or yaw is None:
                raise ValueError("Either all or none of `roll`, `pitch` and "
                                 "`yaw` need to be set.")
            if u is not None or v is not None:
                raise ValueError(
                    "Either set `u` or `v`, or set `roll`, `pitch`"
                    " and `yaw`.")

    @property
    def source(self):
        if isinstance(self._source, Parameter):
            return self._source.value

        return self._source

    @source.setter
    def source(self, value):
        if isinstance(self._source, Parameter):
            self._source.value = value
        else:
            self._source = value

    @property
    def detector(self):
        if isinstance(self._detector, Parameter):
            return self._detector.value

        return self._detector

    @detector.setter
    def detector(self, value):
        if isinstance(self._detector, Parameter):
            self._detector.value = value
        else:
            self._detector = value

    @property
    def roll(self):
        if isinstance(self._roll, Parameter):
            return self._roll.value

        return self._roll

    @roll.setter
    def roll(self, value):
        if isinstance(self._roll, Parameter):
            self._roll.value = value
        else:
            self._roll = value

    @property
    def pitch(self):
        if isinstance(self._pitch, Parameter):
            return self._pitch.value

        return self._pitch

    @pitch.setter
    def pitch(self, value):
        if isinstance(self._pitch, Parameter):
            self._pitch.value = value
        else:
            self._pitch = value

    @property
    def yaw(self):
        if isinstance(self._yaw, Parameter):
            return self._yaw.value

        return self._yaw

    @yaw.setter
    def yaw(self, value):
        if isinstance(self._yaw, Parameter):
            self._yaw.value = value
        else:
            self._yaw = value

    @staticmethod
    def u(r, p, y):
        """Horizontal u-vector in the detector frame."""
        R = StaticGeometry.angles2mat(r, p, y)
        return R.T @ [0, 1, 0]

    @staticmethod
    def v(r, p, y):
        """Vertical v-vector in the detector frame."""
        R = StaticGeometry.angles2mat(r, p, y)
        return R.T @ [0, 0, 1]

    @staticmethod
    def angles2mat(r, p, y) -> np.ndarray:
        return transforms3d.euler.euler2mat(
            r, p, y,
            StaticGeometry.ANGLES_CONVENTION
        )

    @staticmethod
    def mat2angles(mat) -> tuple:
        return transforms3d.euler.mat2euler(
            mat,
            StaticGeometry.ANGLES_CONVENTION
        )

    @classmethod
    def fromDetectorVectors(cls, source, detector, u, v, roll=0., pitch=0.,
                            yaw=0.):
        """Initiate from of detector vectors u, v (in the detector reference
        frame) with intrinsic angles."""

        # convert u,v to global RPY matrix
        n = np.cross(u, v)
        R = np.array([n, u, v])

        # roll this basis
        R_intrinsic = cls.angles2mat(roll, pitch, yaw)
        R_rolled = R_intrinsic.T @ R

        # get global coordinates
        r, p, y = cls.mat2angles(R_rolled)
        return cls(source, detector, r, p, y)

    def own_parameters(self) -> dict:
        params = {}

        if isinstance(self._source, Parameter):
            params['source'] = self._source
        if isinstance(self._detector, Parameter):
            params['detector'] = self._detector
        if isinstance(self._roll, Parameter):
            params['roll'] = self._roll
        if isinstance(self._pitch, Parameter):
            params['pitch'] = self._pitch
        if isinstance(self._yaw, Parameter):
            params['yaw'] = self._yaw

        return params

    def parameters(self) -> list:
        return list(self.own_parameters().values())


class BaseDecorator(StaticGeometry, ABC):
    def __init__(self, decorated_geometry: StaticGeometry):
        self._g = decorated_geometry

    @property
    def decorated_geometry(self):
        return self._g

    @property
    def source(self) -> np.ndarray:
        return self._g.source

    @property
    def detector(self) -> np.ndarray:
        return self._g.detector

    @property
    def roll(self) -> float:
        return self._g.roll

    @property
    def pitch(self) -> float:
        return self._g.pitch

    @property
    def yaw(self) -> float:
        return self._g.yaw

    @abstractmethod
    def parameters(self) -> list:
        # Forcing the user to think about this, however the default would
        # be to return self._g.parameters().
        raise NotImplementedError()


class transform(BaseDecorator):
    """Describes a coordinate transformation to a new orthogonal basis"""

    def __init__(self,
                 geom: StaticGeometry,
                 roll: Any = 0.,
                 pitch: Any = 0.,
                 yaw: Any = 0.):
        """Initialization

        Fill in `None` for parameters that you don't want to use (they'll
        default to 0.), and `0.` for parameters that need to be optimized.

        :param axis:
            An array with the *unit vector* that describes the tilt. Other
            parametrizations (rpy) would be possible, but not implemented.
        :param axis_bounds:
        """
        super().__init__(decorated_geometry=geom)

        self.__roll = roll
        self.__pitch = pitch
        self.__yaw = yaw

    @property
    def transformation_roll(self) -> float:
        if isinstance(self.__roll, Parameter):
            return self.__roll.value
        else:
            return self.__roll

    @property
    def transformation_pitch(self) -> float:
        if isinstance(self.__pitch, Parameter):
            return self.__pitch.value
        else:
            return self.__pitch

    @property
    def transformation_yaw(self) -> float:
        if isinstance(self.__yaw, Parameter):
            return self.__yaw.value
        else:
            return self.__yaw

    def __R(self):
        return StaticGeometry.angles2mat(
            self.transformation_roll,
            self.transformation_pitch,
            self.transformation_yaw,
        )

    def __rpy(self):
        """
        Let `d` be the location of the detector, and `x` be some point in the
        detector frame, and `S` be the linear rotation operator of that frame.
        Then
            x' = R.T(d + S.T x) = R.T d + R.T S.T x = d' + S'.T x
        is the point after rotation with our rotation matrix `R`. In our new
        parametrization `d' + S'x'` we have S'.T = R.T S.T, and
        the r', p' and y' can just be retrieved from those since
        S' = S R
        """
        S = StaticGeometry.angles2mat(self._g.roll, self._g.pitch, self._g.yaw)
        # TODO check if I need to get the RPY from S_prime or S_prime.T
        #    it must be the opposite operation of angles2mat?
        return StaticGeometry.mat2angles(S @ self.__R())

    @property
    def source(self):
        return self.__R().T @ self._g.source

    @property
    def detector(self):
        return self.__R().T @ self._g.detector

    @property
    def roll(self):
        r, p, y = self.__rpy()
        return r

    @property
    def pitch(self):
        r, p, y = self.__rpy()
        return p

    @property
    def yaw(self):
        r, p, y = self.__rpy()
        return y

    def parameters(self) -> list:
        # Adding these parameters will result in parameter duplication amongst
        # geometries that decorate the same underlying geometry.
        # This is not a problem, as we can de-duplicate the parameters before
        # feeding into to the optimization procedure.
        params = self._g.parameters()

        if isinstance(self.__roll, Parameter):
            params.append(self.__roll)
        if isinstance(self.__pitch, Parameter):
            params.append(self.__pitch)
        if isinstance(self.__yaw, Parameter):
            params.append(self.__yaw)

        return params


class shift(BaseDecorator):
    def __init__(self,
                 geom: StaticGeometry,
                 vector: Any = (0., 0., 0.)):
        super().__init__(decorated_geometry=geom)

        if not isinstance(vector, Parameter):
            vector = np.array(vector)
            if len(vector) != 3:
                raise ValueError
        else:
            if len(vector.value) != 3:
                raise ValueError

        self.__vector = vector

    @property
    def vector(self):
        if isinstance(self.__vector, Parameter):
            return self.__vector.value

        return self.__vector

    @property
    def source(self):
        return self._g.source + self.vector

    @property
    def detector(self):
        return self._g.detector + self.vector

    def parameters(self) -> list:
        params = self._g.parameters()
        if isinstance(self.__vector, Parameter):
            params.append(self.__vector)

        return params


def xray_project(geom: StaticGeometry, location: np.ndarray):
    """X-ray projection

    Not sure what consensus is, but I found the geometric method quite
    confusing so I was thinking this would be simple as well.

    Consider the ray going through the source `s` and point `p`:
        r(t) = s + t * (p-s)
    The ray is parametrized by the scalar `t`. `s` and `p` are known vectors.

    Now consider the detector plane equation, an affine linear space:
        0 = d + y*u + z*v
    where `d` is the detector midpoint vector, and `u` and `v` are a orthogonal
    basis for the detector plane. `y` and `z` are again scalars that parametrize
    the plane. Again, `u`, `v` and `d` are known.

    We are now looking to solve _where_ the ray hits the detector plane. This
    is we want to find {t, y, z} that solve
        s + t*(p-s) = d + y*u + z*v,
    or, equivalently,
        t*(s-p) + y*u + z*v = s - d.

    I wouldn't know how to differentiate to a solution a linear system of
    equations, but here we are lucky. We already have an orthogonal basis,
        (u, v, u cross v)
    and hence we know that if we rotate+shift to a basis so that the detector
    is the central point and u=(0,1,0) and v=(0,0,1) the system becomes:
        t*(s-p) + y*(0,1,0) + z*(0,0,1) = s - d;
        t*(s-p) + (0,y,0) + (0,0,z) = s - d.
    Of which the first solution for `t` is free:
        t*(s-p)[0] = (s-d)[0] => t = (s-p)[0]/(s-d)[0]
    Now having `t`, we substitute to get the other two equations:
        => y =  ...
        => z =  ...

    I expect AD to have little trouble differentiating all this.
    """
    R = StaticGeometry.angles2mat(geom.roll, geom.pitch, geom.yaw)

    # get `p-s`, `s-d`, `d` transformed
    p = np.dot(R, location - geom.detector)
    s = np.dot(R, geom.source - geom.detector)

    # solve ray parameter
    t = s[0] / (s - p)[0]

    # get (0, y, z) in the detector basis
    y = s[1] + t * (p - s)[1]
    z = s[2] + t * (p - s)[2]

    return np.array((y, z))


def xray_multigeom_project(geoms, markers: dict):
    """Project all markers with all geometries
    TODO: vectorize
    """

    data = []
    for g in geoms:
        projs = {}
        for id, marker in markers.items():
            v = marker.value if isinstance(marker, MarkerLocation) else marker
            projs[id] = xray_project(g, v)

        data.append(projs)

    return data


class XrayOptimizationProblem:
    def __init__(self, markers, geoms, data):
        self.markers = markers
        self.geoms = geoms
        self.data = np.array(data).flatten()
        # self.data = data

    def params(self):
        """Consistent conversion of markers and geoms to list of parameters"""

        params = []
        for id in sorted(self.markers.keys()):
            params.append(self.markers[id])
        # params = copy.copy(self.markers)

        for g in self.geoms:
            for p in g.parameters():
                params.append(p)

        # remove duplicates, it is important this is done in a consistent way
        # so that when `update()` is called, the same parameters are updated.
        deduped = []
        for p in params:
            if p not in deduped:
                deduped.append(p)

        assert len(deduped) == len(set(params))
        return deduped

    def bounds(self):
        params = self.params()
        return (
            params2ndarray(params, key='min_bound'),
            params2ndarray(params, key='max_bound')
        )

    def update(self, x):
        update_params(self.params(), x)
        return self.geoms, self.markers

    def __call__(self, x: np.ndarray):
        """Optimization call"""
        self.update(x)  # params restore values from `x`

        # only project markers when there is annotated data for that marker
        residuals = []
        for geom, proj in zip(self.geoms, self.data):
            available_annotations = [(self.markers[id].value, p) for id, p in
                                     proj.items()]
            for marker, projected in available_annotations:
                residual = xray_project(geom, marker) - projected
                residuals.append(residual)

        return np.array(residuals).flatten()


def markers_from_leastsquares_intersection(
    geoms,
    data,
    optimizable: bool = False,
    plot: bool = False
):
    """
    What we'll do is infer the position of markers in the volume, by a 
    least-squares intersection of lines that walk from the projected point 
    on the detector to the source location.
    
    https://silo.tips/download/least-squares-intersection-of-lines"""

    if plot:
        import matplotlib.pyplot as plt
        # This import registers the 3D projection, but is otherwise unused.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_box_aspect([150, 150, 40])

    # There is a complication with partially-annotated data that for instance
    # occurs when derived from a tiled scan (not every marker is visible on
    # every scan), or with difficult data, where marker locations are skipped
    # because they could not be accurately annotated.
    # For this, we'll have to assume that data is a partial array, so we'll
    # have to go through the data and see which geometries annotate which data,
    # and if the number of annotations is large enough for a
    # line intersection (n>= 2).
    projections = {}
    for proj, geom in zip(data, geoms):
        for id, pixel in proj.items():
            if not (id in projections):
                projections[id] = []

            # store both the pixel location and associated geometry
            projections[id].append((geom, pixel))

    # are there enough projections?
    for id, proj_point in projections.items():
        if len(proj_point) <= 2:
            raise Exception(f"Unsufficient data for point with id {id}")

    markers = {}
    for id, projected_markers in projections.items():
        R = np.zeros((3, 3))
        q = np.zeros(3)

        # one entity (needle) on multiple projections
        ns = []
        ss = []
        ds = []
        for g, p in projected_markers:  # type: (StaticGeometry, np.ndarray)
            # the projection point in physical space
            R_det = StaticGeometry.angles2mat(g.roll, g.pitch, g.yaw)
            y = g.detector + R_det.T @ [0., p[0], p[1]]
            n = g.source - y  # normal vector
            n /= np.linalg.norm(n)
            Rj = np.identity(3) - np.outer(n, n)
            R += Rj
            q += Rj.dot(y)

            if plot:
                ns.append(y)
                ss.append(g.source)
                ds.append(g.detector)

                k = lambda i: [g.source[i], y[i]]
                ax.plot(k(0), k(1), k(2), 'gray')

        x = np.linalg.pinv(R) @ q
        markers[id] = MarkerLocation(np.array(x), optimize=optimizable)

        if plot:
            ns = np.array(ns).T
            ss = np.array(ss).T
            ds = np.array(ds).T
            ax.scatter(x[0], x[1], x[2], marker='.', s=150)
            ax.scatter(ns[0], ns[1], ns[2], marker='o')
            ax.scatter(ss[0], ss[1], ss[2], marker='x')
            ax.scatter(ds[0], ds[1], ds[2], marker='|')

    if plot:
        import matplotlib.pyplot as plt
        plt.show()

    return markers


def plot_markers(markers):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    xs = [p[0] for p in markers]
    ys = [p[1] for p in markers]
    zs = [p[2] for p in markers]
    ax.scatter(xs, ys, zs)
    plt.show()


def plot_projected_markers(*projected_markers, det=None, det_padding=1.2):
    """

    :param projected_markers:
    :param det:
        If not `None` must have properties `width` and `height`
    :return:
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    from matplotlib.patches import Rectangle

    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    if det is not None:
        plt.xlim(-det_padding * det.width, det_padding * det.width)
        plt.ylim(-det_padding * det.height, det_padding * det.height)

    for set_i, (set, m) in enumerate(zip(projected_markers, ['o', 'x', '*'])):
        ys = [p[0] for k, p in set.items()]
        zs = [p[1] for k, p in set.items()]
        plt.scatter(ys, zs, marker=m)

        for i, (k, p) in enumerate(set.items()):
            ax.annotate(i, (ys[i] + set_i * 10, zs[i]))

    if det is not None:
        rect = Rectangle((-det.width, -det.height),
                         2 * det.width, 2 * det.height,
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
