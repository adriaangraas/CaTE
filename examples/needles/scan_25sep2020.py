import copy
import itertools

import matplotlib.pyplot as plt
import numpy as np
import reflex
from reflex import reco

from cate import annotate
from cate.xray import plot_projected_markers, xray_multigeom_project
from needles.util import astra_reco, astra_residual, geoms_from_interpolation, \
    geoms_from_reflex, \
    needle_data_from_reflex, needle_path_to_location_filename, pixels2coords, \
    plot_astra_volume, plot_residual, run_calibration, \
    run_initial_marker_optimization

"""
The idea is that we can use 3 projections of the needle scan to find the
FleX-ray geometry parameters of a scan, provided that we have good initial
conditions.

What we do here is we take with `flexray_pos2_geoms` an estimate of the geoms
that is obtained after manually calibrating the geoms.

Then with `needles_points`, we take the estimated start and ends of the needles
in the volumes that are obtained with backprojection of the data. These are
the estimated positions of the markers.
"""


class NeedleEntityLocations(annotate.EntityLocations):
    ENTITY_TYPES = ('eye', 'nail', 'ball')
    ENTITY_ORIENTATIONS = ('stake', 'drill')
    ENTITY_LOCATIONS = ('high', 'low')

    @staticmethod
    def nr_entities():
        return len(NeedleEntityLocations.ENTITY_TYPES) \
               * len(NeedleEntityLocations.ENTITY_ORIENTATIONS) \
               * len(NeedleEntityLocations.ENTITY_LOCATIONS)

    @staticmethod
    def get_iter():
        # lst = []
        # for t, o, l in itertools.product(NeedleEntityLocations.ENTITY_TYPES,
        #                                  NeedleEntityLocations.ENTITY_ORIENTATIONS,
        #                                  NeedleEntityLocations.ENTITY_LOCATIONS):
        #     lst.append(f"{t} {o} {l}")
        # return iter(lst)
        return itertools.product(NeedleEntityLocations.ENTITY_TYPES,
                                 NeedleEntityLocations.ENTITY_ORIENTATIONS,
                                 NeedleEntityLocations.ENTITY_LOCATIONS)


# STEP 1: ---------------------------------------------------------------------
# obtaining high-quality positions of markers
# -----------------------------------------------------------------------------
# Find highly accurate needle points from a pre-scan
plot_all = True
plot_step1 = True | plot_all
plot_step2 = True | plot_all
plot_step3 = True | plot_all
prefix = '/bigstore/felix'
# prefix = '/home/adriaan/data'
phantom_dir = f'{prefix}/NeedleCalibration/25Sep2020/pos_2/'
# phantom_dir = '/home/adriaan/data/NeedleCalibration/11Dec2020/source0dec0'
phantom_detector = reco.Reconstruction(phantom_dir).detector()

phantom_offset = 0
# in pos2 projection nrs for annotated markerpoints, there are [0, ..., 3600]
# projs, where 0 and 3600 are not exactly the same, although they might
# supposed to be so it could be a small incremental error that we might safely
# ignore, I don't know
phantom_projs_amount = reflex.nr_projs(phantom_dir) - 1
phantom_calib_nrs = [phantom_offset,
                     phantom_offset + 3600 // 3,
                     phantom_offset + 2 * 3600 // 3]
phantom_reco_downsampling_factor = 5
phantom_reco_nrs = range(0, 3600, phantom_reco_downsampling_factor)
phantom_calib_angles = [i / 3600 * 2 * np.pi for i in phantom_calib_nrs]
phantom_calib_geoms = geoms_from_reflex(phantom_dir, phantom_calib_angles,
                                        parametrization='rotation_stage')
phantom_data = needle_data_from_reflex(phantom_dir,  # annotated data
                                       phantom_calib_nrs,
                                       NeedleEntityLocations,
                                       needle_path_to_location_filename(
                                           phantom_dir),
                                       open_annotator=False)
pixels2coords(phantom_data, phantom_detector)
phantom_markers = run_initial_marker_optimization(
    phantom_calib_geoms,
    # np.array([list(d.values()) for d in phantom_data]),
    phantom_data,
    nr_iters=10,
    plot=False)
# full-angle reconstruction


voxels_x = 300
phantom_vol_id, phantom_vol_geom, phantom_proj_id, phantom_proj_geom = \
    astra_reco(phantom_dir,
               phantom_reco_nrs,
               algo='fdk',
               angles=[i / 3600 * 2 * np.pi for i in phantom_reco_nrs],
               voxels_x=voxels_x
               )
vmin, vmax = -1.5, 1.5  # manually determined value

interpolated_geoms = geoms_from_interpolation(
    tilted_geom=phantom_calib_geoms[0].decorated_geometry,
    interpolation_geoms=phantom_calib_geoms,
    interpolation_nrs=phantom_reco_nrs,
    interpolation_calibration_nrs=phantom_calib_nrs
)
interpolated_phantom_vol_id, interpolated_phantom_vol_geom, interpolated_phantom_proj_id, interpolated_phantom_proj_geom = \
    astra_reco(phantom_dir,
               phantom_reco_nrs,
               algo='fdk',
               geoms=interpolated_geoms,
               voxels_x=voxels_x
               )

if plot_step1:
    # Reproject the groundtruth-volume to see how well the groundtruth volume
    # has been calibrated.
    plot_astra_volume(phantom_vol_id, phantom_vol_geom)
    plot_astra_volume(interpolated_phantom_vol_id, interpolated_phantom_vol_geom)

    # reproject
    proj_id_new = reco.Reconstruction.forward(
        volume_id=phantom_vol_id,
        volume_geom=phantom_vol_geom,
        projection_geom=phantom_proj_geom,
    )

    # residual without calibration (don't know how to subtract on GPU with ASTRA)
    res = reco.Reconstruction.sinogram(phantom_proj_id) - \
          reco.Reconstruction.sinogram(proj_id_new)
    plot_residual(
        np.array(phantom_calib_nrs) // phantom_reco_downsampling_factor,
        res, vmin=vmin, vmax=vmax,
        title='residual: before marker optimization-calibration')
    plt.pause(.001)

    # Now reproject using the calibrated geometry.
    res = astra_residual(phantom_dir, phantom_calib_nrs, phantom_vol_id,
                         phantom_vol_geom,
                         geoms=phantom_calib_geoms)
    plot_residual(range(len(phantom_calib_angles)), res, vmin=vmin, vmax=vmax,
                  title='residual: after marker optimization-calibration')
    plt.pause(.001)

    # Now reproject using the calibrated geometry and interpolated reconstr.
    res = astra_residual(phantom_dir, phantom_calib_nrs,
                         interpolated_phantom_vol_id,
                         interpolated_phantom_vol_geom,
                         geoms=phantom_calib_geoms)
    plot_residual(range(len(phantom_calib_angles)), res, vmin=vmin, vmax=vmax,
                  title='residual: after marker calibration and reinterpolation')
    plt.show()

    # If pre-calibration works well, it showed that even with a
    # "wrong reconstruction" and geometry optmization, you can get slight
    # improvements in the projected residual.

# STEP 2: ---------------------------------------------------------------------
# calibrate a perturbation
# -----------------------------------------------------------------------------

# Now let's see how well we can calibrate a perturbed geometry of which
# we do not have a good estimate.
# In pert2 there are [0, 14] projs, with 0 and 14 the same
#  - Images with 1 blocking needle or more are 0,2,3,6,9,10
# there are 15 images (0, ..., 14), of which 0 and 14 are the same.
# so there is effectively 14 images
# however in pert_0 and pert_2++ there are 24 images
experiment_dir = f'{prefix}/NeedleCalibration/25Sep2020/pert_2++/'
name = experiment_dir.strip('/').split('/')[-1]
experiment_projs_amount = reflex.nr_projs(experiment_dir) - 1  # last eq first
if name == 'pert_0':  # 24 images
    experiment_nrs = [0, 8, 16]  # 0, 4, 8, 16 are annotated
elif name == 'pert_2++':  # 24 images
    # not a triangle!
    experiment_nrs = [0, 4, 8]  # 0, 4, 8 are annotated
elif name == 'pert_2':
    experiment_nrs = [0, 4, 8]
else:
    raise NotImplementedError(f"I have not yet annotated experiment {name}")

experiment_angles = [i / experiment_projs_amount * 2 * np.pi for i in
                     experiment_nrs]

# Load data with annotation
data = needle_data_from_reflex(
    experiment_dir, experiment_nrs, NeedleEntityLocations,
    needle_path_to_location_filename(experiment_dir),
    open_annotator=False)
pixels2coords(data, phantom_detector)

# As a guess for the geometry, in this case we take the geometry of the phantom
geoms = geoms_from_reflex(
    # note `phantom_dir` not `perturb_dir`, as we pretend to not know the geom
    phantom_dir,
    experiment_angles)

# We assume we have the points still from the groundtruth reconstruction
markers = copy.copy(phantom_markers)

# Rotate the found points so that they match
# TODO: this information we don't have in practice! and we'll probably have
#   to do a manual or bruteforce search over possible angles.
# for p in markers:
#     p.optimize = False
    # TODO: implement rotation

# Baseline: how good is the geometry guess: projected phantom points vs data
if plot_step2:
    projection_from_guess = xray_multigeom_project(geoms, markers)
    for d1, d2 in zip(data, projection_from_guess):
        plot_projected_markers(d1, d2, det=phantom_detector)
    plt.show()

# Now perform calibration, and re-evaluate this.
run_calibration(geoms, markers, data)

if plot_step2:
    projection_from_calibration = xray_multigeom_project(geoms, markers)
    for d1, d2 in zip(data, projection_from_calibration):
        plot_projected_markers(d1, d2, det=phantom_detector)
        plt.show()

# STEP 3: ---------------------------------------------------------------------
# validate if found geometry reduces residual on the projections
# -----------------------------------------------------------------------------

# Since the perturbations cannot always be measured in the same perfect
# triangle as that the phantom uses to infer the markers, we have to find the
# phantom geometries that correspond to the `nrs` used in the calibration
# procedure. Those have not been annotated, so are in the current situation
# not profiting from the annotated marker optimization procedure, so may
# exhibit small residual errors that correspond to the FleX-ray miscalibration.
phantom_corresponding_nrs = (np.array(experiment_nrs) / experiment_projs_amount
                             * phantom_projs_amount).astype(np.int)
phantom_corresponding_angles = [i / phantom_projs_amount * 2 * np.pi for i in
                                phantom_corresponding_nrs]
phantom_geoms_for_experiment = geoms_from_reflex(phantom_dir,
                                                 phantom_corresponding_angles)

if plot_step3:
    # Project the phantom using the initial geoms onto experimental projs
    # This should show a big residual shift the experiment is heavily perturbed
    res = astra_residual(experiment_dir, experiment_nrs, phantom_vol_id,
                         phantom_vol_geom,
                         geoms=phantom_geoms_for_experiment)
    plot_residual(range(len(experiment_nrs)), res, vmin=vmin, vmax=vmax,
                  title='residual validation: before experiment')
    plt.pause(.001)

    # Project the phantom using the found geometry onto experimental projs
    # This shoud
    res2 = astra_residual(experiment_dir, experiment_nrs, phantom_vol_id,
                          phantom_vol_geom,
                          geoms=geoms)
    plot_residual(range(len(experiment_nrs)), res2, vmin=vmin, vmax=vmax,
                  title='residual validation: after experiment calibration')
    plt.show()

# If all looks good, we're almost there. Is this "a solution" or the real?

# The problem is that we have only optimized 3 geometries that we had to annotate.
# The reconstruction is build from all geoms, and a 3-angle reconstruction
# will obviously have a huge residual (even with high number of SIRT iters)