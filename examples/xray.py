import numpy as np

from cate.xray import MarkerLocation, StaticGeometry


def rotate_points(points, roll=0., pitch=0., yaw=0.):
    """In-place rotation of points"""

    R = StaticGeometry.angles2mat(roll, pitch, yaw)
    for p in points:
        if isinstance(p, MarkerLocation):
            p.value = R @ p.value
        elif isinstance(p, np.ndarray):
            p[:] = R @ p
        else:
            raise TypeError("Values in `points` have to be `Point` or"
                            " `ndarray`.")

def triangle_geom(src_rad, det_rad):
    gms = []
    for src_a in [0, 2 / 3 * np.pi, 4 / 3 * np.pi]:
        det_a = src_a + np.pi  # opposing
        src = src_rad * np.array([np.cos(src_a), np.sin(src_a), 0])
        det = det_rad * np.array([np.cos(det_a), np.sin(det_a), 0])
        geom = StaticGeometry.fromOrthogonal(
            source=src,
            detector=det,
            # u=np.array([np.cos(det_a +np.pi / 2), np.sin(det_a + np.pi/2), 0]),
            # v=np.array([0, 0, 1])
        )
        gms.append(geom)

    return gms


def cube_points(w=1., d=2., h=4., optimize=False):
    points = [
        [0, 0, 0], [0, d, 0],
        [w, 0, 0], [w, d, 0],
        [0, 0, h], [0, d, h],
        [w, 0, h], [w, d, h],
    ]

    # shift points to center
    for p in points:
        p[0] -= w / 2
        p[1] -= d / 2
        p[2] -= h / 2

    point_objs = []
    for point in points:
        point_objs.append(MarkerLocation(point, optimize))

    return point_objs


def triangle_column_points(rad=4., height=4., start_angle=0., num_angles=3,
                           optimize=False):
    # (0,0) is in the center of the triangle at the base
    angles = np.linspace(start_angle, start_angle + 2 * np.pi,
                         num=num_angles, endpoint=False)

    points = []
    for a in range(len(angles)):
        p = MarkerLocation(
            [rad * np.cos(angles[a]), rad * np.sin(angles[a]), -height / 2],
            optimize)
        points.append(p)
        p = MarkerLocation(
            [rad * np.cos(angles[a]), rad * np.sin(angles[a]), height / 2],
            optimize)
        points.append(p)

    return points
