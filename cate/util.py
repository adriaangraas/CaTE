import numpy as np

from cate.param import ScalarParameter, VectorParameter
from cate.xray import Geometry, transform


def circular_geometry(
    source_position: np.ndarray,
    detector_position: np.ndarray,
    nr_angles: int,
    angle_start: float = 0.,
    angle_stop: float = 2 * np.pi,
    parametrization=None
):
    if parametrization is not None:
        parameters = {
            'source': VectorParameter(source_position),
            'detector': VectorParameter(detector_position),
            'roll': ScalarParameter(None),
            'pitch': ScalarParameter(None),
            'yaw': ScalarParameter(None)
        }
        initial_geom = Geometry(
            source=parameters['source'],
            detector=parameters['detector'],
            roll=parameters['roll'],
            pitch=parameters['pitch'],
            yaw=parameters['yaw'])
    else:
        parameters = {}
        initial_geom = Geometry(
            source=source_position,
            detector=detector_position,
            roll=None,
            pitch=None,
            yaw=None)

    angular_increment = (angle_stop - angle_start) / nr_angles

    geoms = [transform(initial_geom)]
    if parametrization is None:
        g = geoms[0]
        for i in range(1, nr_angles):
            g = transform(g, yaw=angular_increment)
            geoms.append(g)
        geoms = [g.asstatic() for g in geoms]
    elif parametrization == 'constant_rotation':
        # has the disadvantage that it is difficult to interpolate from
        # "rotation_yaw" goes from 0 to angular_incr, and from there
        # stays constant
        rotation_speed = ScalarParameter(angular_increment)
        parameters['rotation_speed'] = rotation_speed
        for i in range(1, nr_angles):
            geoms.append(transform(geoms[-1], yaw=rotation_speed))
    elif parametrization == 'rotation_from_init':
        # a different rotation param for each angle
        for i in range(1, nr_angles):
            angle = ScalarParameter(angular_increment * i)
            parameters[f'angle_{i}'] = angle
            geoms.append(transform(geoms[0], yaw=angle))
    else:
        raise ValueError("Unkown parametrization")

    return geoms, parameters


def geoms_from_interpolation(
    interpolation_geoms,
    interpolation_nrs,
    interpolation_calibration_nrs,
    plot: bool = False,
    method='transforms'
):
    """
    :param interpolation_geoms:
        `transform` geoms at interpolation_calibration_nrs
    :param interpolation_nrs:
        The numbers to interplate to.
    :param interpolation_calibration_nrs:
        The numbers to interpolate from.
    :param plot:
    :return:
    """
    from scipy import interpolate

    def interpolate_var(v, fill_value):
        """Reinterpolate angles (x=interp_calib_nrs, y=angles)"""
        v = np.squeeze(np.array(v))
        f = interpolate.interp1d(
            x=np.array(interpolation_calibration_nrs),
            y=v,
            fill_value=fill_value)

        xnew = interpolation_nrs
        ynew = f(xnew)

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(interpolation_calibration_nrs, v, 'o', xnew, ynew, '-')
            plt.show()

        return ynew

    if method == 'transforms':
        rolls = interpolate_var(
            [g.transformation_roll for g in interpolation_geoms],
            fill_value='extrapolate')
        pitches = interpolate_var(
            [g.transformation_pitch for g in interpolation_geoms],
            fill_value='extrapolate')
        yaws = interpolate_var(
            [g.transformation_yaw for g in interpolation_geoms],
            fill_value='extrapolate')
    elif method == 'statics':
        rolls = interpolate_var([g.roll for g in interpolation_geoms],
                                fill_value='interpolate')
        pitches = interpolate_var([g.pitch for g in interpolation_geoms],
                                  fill_value='interpolate')
        yaws = interpolate_var([g.yaw for g in interpolation_geoms],
                               fill_value='interpolate')
    else:
        raise ValueError

    geoms = []
    for i, (r, p, y) in enumerate(zip(rolls, pitches, yaws)):
        parent_geom = interpolation_geoms[0].decorated_geometry
        rotated_geom = transform(parent_geom,
                                 roll=r,
                                 pitch=p,
                                 yaw=y)
        geoms.append(rotated_geom)

    return geoms


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

    keys = projected_markers[0].keys()
    for set_i, (set, symbol) in enumerate(
        zip(projected_markers, ['o', 'x', '*'])):
        ys = [p[0] for k, p in sorted(set.items())]
        zs = [p[1] for k, p in sorted(set.items())]
        plt.scatter(ys, zs, marker=symbol)

        for i, (k, p) in enumerate(set.items()):
            ax.annotate(i, (ys[i] + set_i * 10, zs[i]))

    if det is not None:
        rect = Rectangle((-det.width / 2, -det.height / 2),
                         det.width, det.height,
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
