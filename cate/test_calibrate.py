import pytest

from fbrct.calibrate import *


@pytest.mark.parametrize(
    "src, det, point, expected_projection, roll, pitch, yaw",
    [
        # simple setup, point on different positions
        ([-1, 0, 0], [1, 0, 0],
         [0, 0, 0], [0, 0],
         0, 0, 0,
         ),
        ([-1, 0, 0], [1, 0, 0],
         [0, 0, 1], [0, 2],
         0, 0, 0,
         ),
        ([-1, 0, 0], [1, 0, 0],
         [0, 1, 0], [2, 0],
         0, 0, 0,
         ),
        ([-1, 0, 0], [1, 0, 0],
         [0, 1, 1], [2, 2],
         0, 0, 0,
         ),
        ([-1, 0, 0], [1, 0, 0],
         [0, 1, 1], [2, 2],
         0, 0, 0,
         ),
        ([-1, 0, 0], [1, 0, 0],
         [.5, 1.5, 1.5], [2, 2],
         0, 0, 0,
         ),

        # move src, det apart but maintain SOD and SDD keeps point on same loc
        ([-4, 0, 0], [2, 0, 0],
         [.5, 1.5, 1.5], [2, 2],
         0, 0, 0,
         ),

        # simple setup, roll 45 degrees clockwise
        # proj (2,2) is now expected at negative rolled position
        # which is (0, 2*sqrt(2))
        ([-1, 0, 0], [1, 0, 0],
         [.5, 1.5, 1.5], [0, 2 * np.sqrt(2)],
         np.pi / 4, 0, 0,
         ),

        # rotated setup along horizontal plane
        ([np.cos(np.pi / 4), np.sin(np.pi / 4), 0],
         [np.cos(5 * np.pi / 4), np.sin(5 * np.pi / 4), 0],
         [np.cos(3 * np.pi / 4), np.sin(3 * np.pi / 4), 1],
         [2, 2],
         0, 0, -np.pi / 4,
         ),

        # rotated setup vertically with pi/3
        ([np.cos(np.pi / 3), 0, np.sin(np.pi / 3)],
         [-np.cos(np.pi / 3), 0, np.sin(-np.pi / 3)],
         [np.cos(np.pi / 6), 1, np.sin(np.pi / 6)],
         [2, 2],
         0, -np.pi / 3, 0,
         ),
    ]
)
def test_xray_project(src, det, point, expected_projection, roll, pitch, yaw):
    # create geometry from angles roll pitch yaw
    geom = StaticGeometry(
        source=np.array(src),
        detector=np.array(det),
        roll=roll,
        pitch=pitch,
        yaw=yaw
    )
    projection = xray_project(geom, np.array(point))
    np.testing.assert_almost_equal(projection, np.array(expected_projection))


@pytest.fixture
def cube_points():
    return cube_points()


@pytest.fixture
def triangle_column_points():
    return triangle_column_points()


def test_geom_rotation_matrix():
    roll = np.pi / 8
    pitch = np.pi / 9
    yaw = np.pi / 10

    # geometry where detector is in position with rpy=0,0,0
    geom = StaticGeometry(
        source=np.array([-1, -2, -3]),
        detector=np.array([3, 2, 1]),
        roll=roll,
        pitch=pitch,
        yaw=yaw
    )

    m = geom.rotation_matrix()
    r, p, y = StaticGeometry.mat2angles(m)

    np.testing.assert_almost_equal(r, roll)
    np.testing.assert_almost_equal(p, pitch)
    np.testing.assert_almost_equal(y, yaw)


@pytest.mark.parametrize('points', ['triangle_column_points', 'cube_points'])
def test_multigeom(geoms, points, request):
    points = request.getfixturevalue(points)

    geoms_true = []
    for g in geoms:
        mat = g.rotation_matrix()
        r, p, y = g.mat2angles(mat)
        geom = StaticGeometry(
            source=g.source + [-1.20, 3.2, 7],
            detector=g.phantom_detector + [-2.0, 2.05, 5],
            roll=r + np.pi / 10,
            pitch=p + np.pi / 10,
            yaw=y + np.pi / 10
        )
        geoms_true.append(geom)

    # perturb one point for later optim
    points_true = []
    for i, p in enumerate(points):
        point = copy.deepcopy(p)
        if i == 0:
            point.value = points[0].value + [-0.02, 0.01, 0.03]
        points_true.append(point)
    data_true = xray_multigeom_op(geoms_true, points_true)

    # make sure to optimize over the points
    for p in points:  # type: Point
        p.optimize = True
        p.bounds[0] = p.value - [0.01] * 3
        p.bounds[1] = p.value + [0.01] * 3

    # points[0].optimize = True

    model = OptimizationProblem(
        points=points,
        geoms=geoms,
        data=data_true,
    )

    import scipy.optimize
    r = scipy.optimize.least_squares(
        fun=model,
        x0=params2ndarray(model.params()),
        bounds=model.bounds(),
        verbose=1,
        jac='3-point'
    )

    geoms_result, points_result = model.update(r.x)

    # plot_scatter3d([p.value for p in points])
    # for d1, d2 in zip(data_true, xray_multigeom_op(geoms, points)):
    #     plot_scatter2d(d1, d2)
    # for d1, d2 in zip(data_true, xray_multigeom_op(geoms_result, points_result)):
    #     plot_scatter2d(d1, d2)

    for i, (g1, g2) in enumerate(zip(geoms_true, geoms_result)):
        print(f"--- GEOM {i} ---")
        print(f"source   : {g1.source} : {g2.source}")
        print(f"detector : {g1.phantom_detector} : {g2.phantom_detector}")
        m1 = g1.rotation_matrix()
        m2 = g2.rotation_matrix()
        r1, p1, y1 = StaticGeometry.mat2angles(m1)
        r2, p2, y2 = StaticGeometry.mat2angles(m2)
        print(f"roll     : {r1} : {r2}")
        print(f"pitch    : {p1} : {p2}")
        print(f"yaw      : {y1} : {y2}")

        decimal_accuracy = 3
        np.testing.assert_almost_equal(g1.source, g2.source, decimal_accuracy)
        np.testing.assert_almost_equal(g1.phantom_detector, g2.phantom_detector,
                                       decimal_accuracy)
        np.testing.assert_almost_equal(r1, r2, decimal_accuracy)
        np.testing.assert_almost_equal(p1, p2, decimal_accuracy)
        np.testing.assert_almost_equal(y1, y2, decimal_accuracy)

