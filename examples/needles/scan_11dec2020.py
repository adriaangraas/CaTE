import reflex
from reflex import reco
from reflex.motor import Corrections

from cate import annotate
from needles.util import *

"""
Setting file notes
 - It looks like we have a different `rot_obj` acceleration in the dec-50 scan.
 - `ver_tube` and `ver_obj` seem to be extremely minor, so probably are correct.

adriaan[~/data/NeedleCalibration/11Dec2020]$ diff source0dec0/scan\ settings.txt source0dec-50/scan\ settings.txt 
< rot_obj : -0.409904 ; speed : 0.666717 ; acceleration : 22.500000
> rot_obj : -0.419782 ; speed : 0.666717 ; acceleration : 11.250000 <- !!!

< PI_Y : -0.001588 ; speed : 10.000000 ; acceleration : 5.000000
< ver_obj : 276.220131 ; speed : 37.500000 ; acceleration : 125.000000
< ver_tube : 103.777351 ; speed : 37.500000 ; acceleration : 125.000000
< ver_det : 98.140717 ; speed : 37.500000 ; acceleration : 125.000000
---
> PI_Y : -0.001596 ; speed : 10.000000 ; acceleration : 5.000000
> ver_obj : 276.217613 ; speed : 37.500000 ; acceleration : 125.000000
> ver_tube : 103.781166 ; speed : 37.500000 ; acceleration : 125.000000
> ver_det : 48.121986 ; speed : 37.500000 ; acceleration : 125.000000

adriaan[~/data/NeedleCalibration/11Dec2020]$ diff source0dec0/scan\ settings.txt source0dec+50/scan\ settings.txt 
1c1
< PI_Y : -0.001588 ; speed : 10.000000 ; acceleration : 5.000000
< ver_obj : 276.220131 ; speed : 37.500000 ; acceleration : 125.000000
< ver_tube : 103.777351 ; speed : 37.500000 ; acceleration : 125.000000
< ver_det : 98.140717 ; speed : 37.500000 ; acceleration : 125.000000
---
> PI_Y : -0.001591 ; speed : 10.000000 ; acceleration : 5.000000
> ver_obj : 276.218834 ; speed : 37.500000 ; acceleration : 125.000000
> ver_tube : 103.779869 ; speed : 37.500000 ; acceleration : 125.000000

Planned methodic:
1. annotate inidividual detector frames of different directories
   - I don't expect the choice of angles to really matter 
2. formulate a joint geometry in which there is
   - 1 source position parameter (we're pretty sure this is static)
   - 1 detector position parameter for each scan (initialized from settings)
   - 1 tilt parameter for each scan
   - 1 angular rotation parameter for each frame
     we are not linking the rotation parameter of individual scans, because
     there may likely be a speed difference of the rotation table due to the
     different acceleration settings
3. enhance the code so that it is okay if marker points are not annotated, or
   missing from the detector frame, that they are not weighed into the objective
4. ???
5. profit
"""


def stitched_reco(dir, stitching_dirs):
    offsets = [[-670, 4], [670, -4]]
    phantom_reco_nrs = range(0, 3600, 10)
    phantom_reco_angles = [i / 3600 * 2 * np.pi for i in phantom_reco_nrs]

    rec = reflex.reco.StitchedReconstruction(
        dir,
        det_subsampling=2,
        stitching_paths=stitching_dirs,
        stitching_offsets=offsets,
        proj_range=phantom_reco_nrs
    )

    sino = rec.load_sinogram()
    vectors = rec.geom(phantom_reco_angles)
    sino_id, proj_geom = rec.sino_gpu_and_proj_geom(
        sino,
        vectors,
        rec.detector()
    )

    vol_id, vol_geom = rec.backward(
        sino_id, proj_geom, voxels_x=300, algo='fdk', iters=200)

    # phantom_offset = 800
    # phantom_nrs = [phantom_offset,
    #                phantom_offset + 3600 // 3,
    #                phantom_offset + 2 * 3600 // 3]

    plot_astra_volume(vol_id, vol_geom, from_side=True)
    plt.figure()
    plt.pause(1)
    plt.show()

    rec.clear()


def geoms_from_tiling_scan_reflex(tiling_dirs, angles):
    """Custom geometry build from combining FleX-ray geometries

    :param tiling_dirs: list
        A list of directories. The source parameter will be taken from the
        first directory.
    :param angles: list
        A list of lists. The first dim is the directory, the second dim the
        angles.
    """
    import reflex

    for angles_in_dir in angles:
        for a in angles_in_dir:
            assert 0 <= a <= 2 * np.pi

    all_geoms = []

    sett = reflex.Settings.from_path(tiling_dirs[0])
    # corr = Corrections()
    # corr.force_tra_alignment = True
    motor_geom = reflex.motor_geometry(sett, corrections=True, verbose=False)
    # describe a tilt of the rotation stage, which doesn't require yaw
    # (rotation about z-axis)
    motor_geom = reflex.centralize(motor_geom)
    initial_geom = xray.StaticGeometry(
        source=VectorParameter(motor_geom.tube_position),
        detector=VectorParameter(motor_geom.detector_position),
        roll=ScalarParameter(None),
        pitch=ScalarParameter(None),
        yaw=ScalarParameter(None),
    )

    for d, d_angles in zip(tiling_dirs, angles):
        geoms = []

        # move the detector to new location
        # the default behavior of reflex is to correct vertical alignment
        # we stop that here, and only correct alignment in the traverse (y) ax
        corr = Corrections()
        corr.force_tra_alignment = True
        corr.force_ver_alignment = False
        tmp_sett = reflex.Settings.from_path(d)
        tmp_geom = reflex.motor_geometry(tmp_sett,
                                         verbose=False,
                                         corrections=corr)
        tmp_geom = reflex.centralize(tmp_geom)

        # start each tiling scan with a new initial geometry
        initial_geom_copy = xray.StaticGeometry(
            source=initial_geom.own_parameters()['source'],
            detector=VectorParameter(tmp_geom.detector_position),
            roll=initial_geom.own_parameters()['roll'],
            pitch=initial_geom.own_parameters()['pitch'],
            yaw=initial_geom.own_parameters()['yaw'],
        )

        # Individual tilt for every directory, because tilt depends on the
        # uncontrolled rotation of the stage
        # Note: I chose to not trust that the `rot_obj` value is consistent
        # between scans, but that could of course be the case. I doubt it.
        tilted_geom = xray.transform(initial_geom_copy,
                                     roll=ScalarParameter(0.),
                                     pitch=ScalarParameter(0.))

        for angle in d_angles:
            rotated_geom = xray.transform(
                tilted_geom, yaw=ScalarParameter(angle))
            geoms.append(rotated_geom)

        all_geoms.append(geoms)

    return all_geoms


def astra_tiled_reco(
    tiling_paths,
    nrs,
    algo='fdk',
    voxels_x=300,
    angles=None,
    geoms=None,
    iters: int = 250,
    **kwargs
):
    from reflex import reco

    rec = reco.TiledReconstruction(
        tiling_paths=tiling_paths,
        tiling_volume_scaling_factor_height=2,
        tiling_volume_scaling_factor_width=1,
        proj_ranges=nrs,
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


def astra_tiled_residual(tiling_paths,
                         all_nrs,
                         vol_id,
                         vol_geom,
                         angles=None,
                         geoms=None):
    from reflex import reco

    rec = reco.TiledReconstruction(
        tiling_paths,
        tiling_volume_scaling_factor_height=2,
        tiling_volume_scaling_factor_width=1,
        proj_ranges=all_nrs
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


class DelftNeedleEntityLocations(annotate.EntityLocations):
    """
    A bit cryptic, but the encoding is based on the 4 vertical sequences
    of needles glued onto the column, from top to bottom:
        BBB = ball ball ball
        BEE = ball eye eye
        EB = eye ball
        B = ball
    Then with `top`, `middle` of `bottom` I encode which of the three needles
    it is, and also provide additional redundant information for convenience.
    Note however that the vertical description does not tell how far up the
    the needle is in the column, only its relative position. There does not
    seem to be good horizontal alignment, so its difficult to segment the
    column in annotated layers.
    """
    ENTITIES = (
        'BBB  top ball stake',
        'BBB  middle ball stake',
        'BBB  bottom ball stake',
        'BEE  top ball drill',
        'BEE  middle eye drill',
        'BEE  bottom eye drill',
        'EB  top eye stake',
        'EB  bottom ball drill',
        'B  ball drill',
    )

    @staticmethod
    def nr_entities():
        return len(DelftNeedleEntityLocations.ENTITIES)

    @staticmethod
    def get_iter():
        return iter(DelftNeedleEntityLocations.ENTITIES)


if __name__ == '__main__':
    prefix = '/bigstore/felix'
    # prefix = '/home/adriaan/data'
    phantom_dir = f'{prefix}/NeedleCalibration/11Dec2020/source0dec0'
    phantom_tiling_dirs = [
        f'{prefix}/NeedleCalibration/11Dec2020/source0dec+50',
        f'{prefix}/NeedleCalibration/11Dec2020/source0dec-50']

    phantom_dirs = [phantom_dir] + phantom_tiling_dirs

    plot_step0 = False
    plot_step1 = True

    # STEP 1: ---------------------------------------------------------------------
    # check out a stitched reco of the phantom
    # -----------------------------------------------------------------------------
    if plot_step0:
        stitched_reco(phantom_dir, phantom_tiling_dirs)

    # STEP 1: ---------------------------------------------------------------------
    # obtaining high-quality positions of markers
    # -----------------------------------------------------------------------------
    phantom_detector = reco.Reconstruction(phantom_dir).detector()

    phantom_projs_amount = reflex.nr_projs(phantom_dir) - 1
    phantom_calib_nrs = [
        [0,  # dec0
         3600 // 3,
         2 * 3600 // 3],
        [0,  # dec+50
         3600 // 3,
         2 * 3600 // 3],
        [0,  # dec-50
         3600 // 3,
         2 * 3600 // 3],
    ]
    phantom_calib_angles = [[i / 3600 * 2 * np.pi for i in d] for d in
                            phantom_calib_nrs]
    phantom_calib_geoms = geoms_from_tiling_scan_reflex(
        phantom_dirs,
        phantom_calib_angles)
    phantom_calib_geoms_flat = [g for d in phantom_calib_geoms for g in d]
    phantom_data = [
        needle_data_from_reflex(d,  # annotated data
                                nrs,
                                DelftNeedleEntityLocations,
                                needle_path_to_location_filename(d),
                                open_annotator=False)
        for d, nrs in zip(phantom_dirs, phantom_calib_nrs)
    ]
    phantom_data_flat = [p for d in phantom_data for p in d]

    for d in phantom_data:
        pixels2coords(d, phantom_detector)

    # flattening the lists = treating angles and tiling equally as projections
    markers = run_initial_marker_optimization(
        phantom_calib_geoms_flat,
        phantom_data_flat,
        nr_iters=10, plot=False)
    # markers_from_leastsquares_intersection(
    #     phantom_calib_geoms_flat,
    #     phantom_data_flat,
    #     optimizable=False,
    #     plot=False)

    np.save('markers_11dec2020_calibrated_14jan2021.npy', markers)

    # full-angle reconstruction
    # NOTE: must allow int. division by calibration nrs to plot residuals!
    # For instance, if there are 3600 angles, and calibration occurs on angles
    # with numbers 0, 1200, 2400 then downsampling may be 30 but not 36, while
    # both are divisors of 3600
    phantom_reco_downsampling_factor = 5  # * 2 * 2
    phantom_reco_nrs = [range(0, 3600, phantom_reco_downsampling_factor)] \
                       * len(phantom_dirs)
    phantom_reco_angles = [[i / 3600 * 2 * np.pi for i in phantom_reco_nrs[0]]] \
                          * len(phantom_dirs)  # TODO: make nicer

    interpolated_geoms = []
    for i in range(len(phantom_dirs)):
        interpolated_geoms.append(geoms_from_interpolation(
            tilted_geom=phantom_calib_geoms[i][0].decorated_geometry,
            interpolation_geoms=phantom_calib_geoms[i],
            interpolation_nrs=phantom_reco_nrs[i],
            interpolation_calibration_nrs=phantom_calib_nrs[i]))
    interpolated_geoms_flat = [g for d in interpolated_geoms for g in d]

    voxels_x = 500
    phantom_vol_id, phantom_vol_geom, phantom_proj_id, phantom_proj_geom = \
        astra_tiled_reco(phantom_dirs,
                         phantom_reco_nrs,
                         algo='fdk',
                         voxels_x=voxels_x,
                         # geoms=interpolated_geoms_flat,
                         angles=phantom_reco_angles
                         )
    interpolated_phantom_vol_id, interpolated_phantom_vol_geom, interpolated_phantom_proj_id, interpolated_phantom_proj_geom = \
        astra_tiled_reco(phantom_dirs,
                         phantom_reco_nrs,
                         algo='fdk',
                         voxels_x=voxels_x,
                         geoms=interpolated_geoms_flat)

    vmin, vmax = -1.5, 1.5  # manually determined value

    if plot_step1:
        # Reproject the groundtruth-volume to see how well the groundtruth volume
        # has been calibrated.
        plot_astra_volume(phantom_vol_id,
                          phantom_vol_geom, from_side=True)
        plot_astra_volume(interpolated_phantom_vol_id,
                          interpolated_phantom_vol_geom, from_side=True)

        # reproject
        phantom_proj_id_reproject = reco.Reconstruction.forward(
            volume_id=phantom_vol_id,
            volume_geom=phantom_vol_geom,
            projection_geom=phantom_proj_geom,
        )

        # full residual without calibration (don't know how to subtract on GPU with ASTRA)
        res = reco.Reconstruction.sinogram(phantom_proj_id) - \
              reco.Reconstruction.sinogram(phantom_proj_id_reproject)
        for i in range(3):
            corresponding_nrs = (
                (i * phantom_projs_amount + np.array(phantom_calib_nrs[i]))
                / phantom_reco_downsampling_factor
            ).astype(np.int)
            plot_residual(corresponding_nrs, res, vmin=vmin, vmax=vmax,
                          title=f'residuals {i}: before marker optimization-calibration')
            plt.pause(.001)

        # Now reproject using the calibrated geometry.
        res = astra_tiled_residual(phantom_dirs,
                                   phantom_calib_nrs,
                                   phantom_vol_id,
                                   phantom_vol_geom,
                                   geoms=phantom_calib_geoms_flat)
        for i in range(3):
            plot_residual(range(i * 3, i * 3 + 3),  # TODO
                          res, vmin=vmin, vmax=vmax,
                          title=f'residuals {i}: after marker optimization-calibration')
            plt.pause(.001)

        # Now reproject using the calibrated geometry and interpolated reconstr.
        res = astra_tiled_residual(phantom_dirs,
                                   phantom_calib_nrs,
                                   interpolated_phantom_vol_id,
                                   interpolated_phantom_vol_geom,
                                   geoms=phantom_calib_geoms_flat)
        for i in range(3):
            plot_residual(range(i * 3, i * 3 + 3),  # TODO
                          res, vmin=vmin, vmax=vmax,
                          title=f'residuals {i}: after marker calibration and reinterpolation')

        plt.show()
