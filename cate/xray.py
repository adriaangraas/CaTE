import copy

import numpy as np
import transforms3d

from cate.param import ScalarParameter, VectorParameter, \
    params2ndarray, update_params


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
        source: np.ndarray,
        detector: np.ndarray,
        detector_props: Detector,
        u: np.ndarray = None,
        v: np.ndarray = None,
        roll=0.,
        pitch=0.,
        yaw=0.,
        source_bounds=None,
        detector_bounds=None,
    ):
        """Initialialize geometry from source and detector

        The detector frame are the three axis in the detector point where
        the vertical axis is always perfectly upward [0, 0, 1], but the
        in the detector point.
        The u and v that span the detector plane are chosen to be

        """
        self.source_param = VectorParameter(source,
                                            bounds=source_bounds, optimize=True)
        self.detector_param = VectorParameter(detector,
                                              bounds=detector_bounds, optimize=True)
        self.detector_props = detector_props

        if u is None or v is None:
            n = self.source - self.detector  # in global frame coordinates
            n /= np.linalg.norm(n)

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
            roll, pitch, yaw = self.mat2angles(mat)

        self.roll_param = ScalarParameter(roll)
        self.pitch_param = ScalarParameter(pitch)
        self.yaw_param = ScalarParameter(yaw)

    def rotation_matrix(self):
        """Transformation matrix to turn a point in the detector reference frame
        into an actual point on the detector (incorporting RPY)."""
        R = self.angles2mat(self.roll, self.pitch, self.yaw)
        return R.T

    @property
    def source(self):
        return self.source_param.value

    @property
    def detector(self):
        return self.detector_param.value

    @property
    def roll(self):
        return self.roll_param.value

    @property
    def pitch(self):
        return self.pitch_param.value

    @property
    def yaw(self):
        return self.yaw_param.value

    @property
    def u(self):
        """Horizontal u-vector in the detector frame."""
        R = self.angles2mat(self.roll, self.pitch, self.yaw)
        return R.T @ [0, 1, 0]

    @property
    def v(self):
        """Vertical v-vector in the detector frame."""
        R = self.angles2mat(self.roll, self.pitch, self.yaw)
        return R.T @ [0, 0, 1]

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

    def parameters(self):
        return [self.source_param,
                self.detector_param,
                self.roll_param,
                self.pitch_param,
                self.yaw_param]


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
    R = geom.rotation_matrix()

    # get `p-s`, `s-d`, `d` transformed
    p = np.dot(R.T, location - geom.detector)
    s = np.dot(R.T, geom.source - geom.detector)

    # solve ray parameter
    t = s[0] / (s - p)[0]

    # get (0, y, z) in the detector basis
    y = s[1] + t * (p - s)[1]
    z = s[2] + t * (p - s)[2]

    return np.array((y, z))


def xray_multigeom_project(geoms, locations):
    """TODO: vectorize"""

    data = []
    for g in geoms:
        projs = []
        for p in locations:
            v = p.value if isinstance(p, MarkerLocation) else p
            projs.append(xray_project(g, v))

        data.append(projs)

    return data


class XrayOptimizationProblem:
    def __init__(self, markers: list, geoms, data):
        self.markers = markers
        self.geoms = geoms
        self.data = np.array(data).flatten()

    def params(self):
        # prevents (repetitive) adding to the points list
        params = copy.copy(self.markers)
        for g in self.geoms:
            for p in g.parameters():
                params.append(p)

        return params

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
        self.update(x)  # param restore values from `x`
        projs = xray_multigeom_project(self.geoms, self.markers)
        projs = np.array(projs).flatten()
        return projs - self.data


def markers_from_leastsquares_intersection(
    geoms,
    data,
    plot:
    bool = False,
    optimizable: bool = False
):
    """https://silo.tips/download/least-squares-intersection-of-lines"""

    if plot:
        import matplotlib.pyplot as plt
        # This import registers the 3D projection, but is otherwise unused.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_box_aspect([150, 150, 40])

    volume_points = []

    for i in range(data.shape[1]):
        R = np.zeros((3, 3))
        q = np.zeros(3)

        # one entity (needle) on multiple projections
        ns = []
        ss = []
        ds = []
        for g, p in zip(geoms,
                        data[:, i]):  # type: (StaticGeometry, np.ndarray)
            # the projection point in physical space
            y = g.detector + g.rotation_matrix() @ [0., p[0], p[1]]
            # normal vector
            n = g.source - y
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
        volume_points.append(MarkerLocation(np.array(x), optimize=optimizable))

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

    return volume_points


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


def plot_projected_markers(*projected_markers, det=None):
    """

    :param projected_markers:
    :param det: If not `None` must have properties `width` and `height`
    :return:
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    from matplotlib.patches import Rectangle

    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    if det is not None:
        plt.xlim(-1.2 * det.width, 1.2 * det.width)
        plt.ylim(-1.2 * det.height, 1.2 * det.height)

    for set_i, (set, m) in enumerate(zip(projected_markers, ['o', 'x', '*'])):
        ys = [p[0] for p in set]
        zs = [p[1] for p in set]
        plt.scatter(ys, zs, marker=m)

        for i, txt in enumerate(set):
            ax.annotate(i, (ys[i]+set_i*10, zs[i]))

    if det is not None:
        rect = Rectangle((-det.width, -det.height), 2 * det.width, 2 * det.height,
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


