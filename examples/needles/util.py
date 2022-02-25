import copy
import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pq
import scipy.optimize

from cate import xray
from cate.annotate import Annotator
from cate.param import ScalarParameter, VectorParameter, params2ndarray
from cate.xray import Detector, XrayOptimizationProblem, \
    markers_from_leastsquares_intersection


def needle_data_from_reflex(dir,
                            nrs,
                            entity_locations_class,
                            fname='needle_locations.npy',
                            open_annotator=False) -> list:
    """(Re)store marker projection coordinates from annotations

    :return list
        List of `dict`, each dict being a projection angle, and each item
        from the dictionary is a key-value pair of identifier and pixel
        location."""
    import reflex

    if open_annotator:
        projs = reflex.projs(dir, nrs)
        for nr, proj in zip(nrs, projs):
            loc = entity_locations_class(fname, nr)
            Annotator(loc, proj)

    data = []
    for nr in nrs:
        loc = entity_locations_class(fname, nr)
        l = loc.locations()

        # for backwards compatibility (I used tuples before)
        for id, pixel in l.items():
            if isinstance(l[id], tuple):
                l[id] = list(pixel)

        data.append(l)

    return data


def geoms_from_reflex(dir: str, angles, parametrization='default'):
    """Load CATE X-ray geometries from FleX-ray descriptions, using Reflex

    :param parametrization
        The way of initializing CATE geometries. If 'default', then each
        angle from the FleX-ray geometry will have unique `source` and `detector`
        parameters, as well as `roll`, `pitch`, and `yaw`.
        When the `parametrization` is 'rotation_stage', the geometry is encoded
        as an initial static geometry, and all the subsequent geometries are
        parametrized as a rotation of the initial geometry.
    """
    import reflex

    sett = reflex.Settings.from_path(dir)
    motor_geom = reflex.motor_geometry(sett, verbose=False)

    if parametrization == 'rotation_stage':
        geoms = []

        # describe a tilt of the rotation stage, which doesn't require yaw
        # (rotation about z-axis)
        motor_geom = reflex.centralize(motor_geom)
        initial_geom = xray.Geometry(
            source=VectorParameter(motor_geom.tube_position),
            detector=VectorParameter(motor_geom.detector_position),
            roll=ScalarParameter(None),
            pitch=ScalarParameter(None),
            yaw=ScalarParameter(None))
        tilted_geom = xray.transform(initial_geom,
                                     roll=ScalarParameter(0.),
                                     pitch=ScalarParameter(0.))

        # describe as a rotation of `angle` from the initial point
        # the alternative could be to chain-describe the geometries, not sure
        # if that is better
        for i, angle in enumerate(angles):
            rotated_geom = xray.transform(
                tilted_geom,
                yaw=ScalarParameter(angle))
            geoms.append(rotated_geom)

    elif parametrization == 'default':
        reflex_geoms = reflex.circular_geometry(sett,
                                                motor_geom,
                                                angles)
        geoms = []
        for nr, rg in reflex_geoms.to_dict().items():
            rg = reflex.centralize(rg)
            geoms.append(xray.Geometry(
                source=VectorParameter(rg.tube_position),
                detector=VectorParameter(rg.detector_position),
                roll=ScalarParameter(None),
                pitch=ScalarParameter(None),
                yaw=ScalarParameter(None)))
    else:
        raise ValueError

    return geoms


def plot_astra_volume(vol_id, vol_geom, points: Any = False, from_side=False):
    from reflex import reco

    volume = reco.Reconstruction.volume(vol_id)
    if from_side:
        volume = np.transpose(volume, [2, 1, 0])

    if points:
        voxel_x_size = (vol_geom['option']['WindowMaxX'] -
                        vol_geom['option']['WindowMinX']) / vol_geom[
                           'GridRowCount']
        voxel_y_size = (vol_geom['option']['WindowMaxY'] -
                        vol_geom['option']['WindowMinY']) / vol_geom[
                           'GridColCount']
        voxel_z_size = (vol_geom['option']['WindowMaxZ'] -
                        vol_geom['option']['WindowMinZ']) / vol_geom[
                           'GridSliceCount']

        for point in points:
            p = point.value
            point_coords = np.array([-p[2], p[0], p[1]])

            # TODO: not sure if this order is correct
            #   because number of voxels in each direction is the same
            point_coords /= np.array(
                [voxel_x_size, voxel_y_size, voxel_z_size])
            # TODO: not sure if this order is correct
            #   because number of voxels in each direction is the same
            point_coords += np.array([
                vol_geom['GridRowCount'] / 2,
                vol_geom['GridColCount'] / 2,
                vol_geom['GridSliceCount'] / 2]).astype(np.int)

            point_coords = np.round(point_coords).astype(np.int)
            r = volume.shape[0] // 100
            for x_, y_, z_ in itertools.product(
                *(range(s - r, s + r) for s in point_coords)):
                if np.linalg.norm(
                    np.subtract(point_coords, [x_, y_, z_])) < r:
                    volume[x_, y_, z_] += 0.5

        # Maximum Intensity Projections
        mip_x = np.max(volume, axis=0)
        plt.figure()
        plt.imshow(mip_x)
        plt.figure()
        mip_x = np.max(volume, axis=1)
        plt.imshow(mip_x)
        plt.figure()
        mip_x = np.max(volume, axis=2)
        plt.imshow(mip_x)
        plt.show()

    pq.image(volume)
    plt.show()


def astra_reco(
    proj_path,
    nrs,
    algo='fdk',
    voxels_x=300,
    angles=None,
    geoms=None,
    iters: int = 250,
    **kwargs
):
    """
    Ordinary ASTRA reconstruction using FleX-ray files with Reflex

    Assuming data is from one full rotation.

    :param proj_path:
    :param nrs: which projections to use
    :param algo:
    :param voxels_x:
    :param geoms: if own `xray.StaticGeometry` are given, these will be
        preferred over the geometries that are inferred from the FleX-ray
        files
    :param plot_volume:
    :param plot_residual_nrs_inds: Make an additional
    :param title:
    :param iters: number of iterations for `algo`, if it is iterative
    :return:
    """
    from reflex import reco

    rec = reco.Reconstruction(
        path=proj_path,
        proj_range=nrs,
        **kwargs
    )

    sinogram = rec.load_sinogram()

    if geoms is None:
        # take geoms as given in the FleX-ray files
        vectors = rec.geom(angles)
    else:
        # convert input `geoms` to ASTRA vectors, take detector from
        # FleX-ray settings
        vectors = np.array([geom2astravec(g, rec.detector()) for g in geoms])

    sino_id, proj_geom = rec.sino_gpu_and_proj_geom(
        sinogram,
        vectors,
        rec.detector()
    )

    vol_id, vol_geom = rec.backward(
        sino_id, proj_geom, voxels_x=voxels_x, algo=algo, iters=iters)

    return vol_id, vol_geom, sino_id, proj_geom


def astra_residual(projs_path, nrs, vol_id, vol_geom, angles=None, geoms=None):
    """Projects then substracts a volume with `vol_id` and `vol_geom`
    onto projections from `projs_path` with `nrs`.

    Using `geoms` or `angles`. If geoms are perfect, then the residual will be
    zero. Otherwise it will show some geometry forward-backward mismatch.
    """
    from reflex import reco

    rec = reco.Reconstruction(
        path=projs_path,
        proj_range=nrs
    )

    detector = rec.detector()
    if geoms is None:
        if angles is None:
            raise ValueError("Either supply `geoms` or `angles`.")

        vectors = rec.geom(angles)
    else:
        vectors = np.array([geom2astravec(g, rec.detector()) for g in geoms])

    sino_id, proj_geom = rec.sino_gpu_and_proj_geom(
        0.,  # zero-sinogram
        vectors,
        detector
    )

    proj_id = rec.forward(
        volume_id=vol_id,
        volume_geom=vol_geom,
        projection_geom=proj_geom,
    )
    return rec.load_sinogram() - rec.sinogram(proj_id)


def run_calibration(geoms, markers, data, method='trf',
                    loss='huber', verbose=2, max_nfev=None):
    """In-place optimization of `geoms` and `points` using `data`

    :param geoms:
    :param markers:
    :param data:
    :param plot_dets:
    :param verbose:
    :return:
    """

    # `geoms` and `markers` will be optimized in-place, so we'd make a backup here
    # for later reference.
    geoms_initial = copy.deepcopy(geoms)
    # points_initial = copy.deepcopy(markers)

    # make sure to optimize over the points
    # for p in points:  # type: Point
    #     p.optimize = False
    #     p.bounds[0] = p.value - [1 * error_magnitude_points] * 3
    #     p.bounds[1] = p.value + [1 * error_magnitude_points] * 3

    problem = XrayOptimizationProblem(
        markers=markers,
        geoms=geoms,
        data=data,
        use_multiprocessing=False
    )

    r = scipy.optimize.least_squares(
        fun=problem,
        x0=params2ndarray(problem.params()),
        bounds=problem.bounds(),
        verbose=verbose,
        method=method,
        tr_solver='exact',
        loss=loss,
        jac='3-point',
        max_nfev=max_nfev
    )
    geoms_calibrated, markers_calibrated = problem.update(r.x)

    np.set_printoptions(precision=4, suppress=True)
    if verbose >= 2:
        for i, (g1, g2) in enumerate(zip(geoms_initial, geoms_calibrated)):
            print(f"--- GEOM {i} ---")
            print(f"source   : {g1.source} : {g2.source}")
            print(f"detector : {g1.detector} : {g2.detector}")
            print(f"roll     : {g1.roll} : {g2.roll}")
            print(f"pitch    : {g1.pitch} : {g2.pitch}")
            print(f"yaw      : {g1.yaw} : {g2.yaw}")

    for i, (g1, g2) in enumerate(zip(geoms_initial, geoms_calibrated)):
        try:
            decimal_accuracy = 3
            np.testing.assert_almost_equal(g1.source, g2.source,
                                           decimal_accuracy)
            np.testing.assert_almost_equal(g1.detector, g2.detector,
                                           decimal_accuracy)
            np.testing.assert_almost_equal(g1.roll, g2.roll, decimal_accuracy)
            np.testing.assert_almost_equal(g1.pitch, g2.pitch,
                                           decimal_accuracy)
            np.testing.assert_almost_equal(g1.yaw, g2.yaw, decimal_accuracy)
        except:
            pass


def run_initial_marker_optimization(geoms, data, nr_iters: int = 20,
                                    plot=False, **kwargs):
    """Find points of the phantom that we have because of a high-resolution
    prescan.
    """
    # Since the prescan is not perfect, we do a simple optimization process
    # over the prescan parameters.
    # Repeating optimizing over the "ground-truth geometry"
    assert nr_iters > 0
    for i in range(nr_iters):
        # Get Least-Square optimal points by analytically backprojection the
        # marker locations, which is a LS intersection of lines.
        markers = markers_from_leastsquares_intersection(
            geoms, data,
            optimizable=False,
            plot=plot,
        )
        # Then find the optimal geoms given the `points` and `data`, in-place.
        run_calibration(
            geoms,
            markers,
            data,
            **kwargs,
        )

    # noinspection PyUnboundLocalVariable
    return markers


def plot_residual(inds, res, vmin=None, vmax=None, title=None):
    fig, axs = plt.subplots(nrows=1, ncols=3)
    if title is not None:
        plt.title(title)

    for i, ind in enumerate(inds):
        im = axs[i].imshow(res[:, ind], vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i])


def needle_path_to_location_filename(path):
    s = path.strip("/").split("/")
    return f'needle_{s[-2]}_{s[-1]}_locations.npy'
