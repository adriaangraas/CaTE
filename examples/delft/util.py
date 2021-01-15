import numpy as np
from fbrct.loader import load_referenced_projs_from_fulls

from cate import xray
from cate.annotate import Annotator
from cate.param import ScalarParameter, VectorParameter
from needles.util import geom2astravec


def _load_median_projs(path, fulls_path, t):
    p = load_referenced_projs_from_fulls(path,
                                         fulls_path,
                                         t_range=range(t, t + 1),
                                         reference_method='median')
    return np.median(p, axis=0)


def annotated_data(
    projs_path,
    fulls_path,
    t,
    entity_locations_class,
    fname,
    open_annotator=False) -> list:
    """(Re)store marker projection coordinates from annotations

    :return list
        List of `dict`, each dict being a projection angle, and each item
        from the dictionary is a key-value pair of identifier and pixel
        location."""

    if open_annotator:
        projs = _load_median_projs(projs_path, fulls_path, t)
        for cam_nr, proj in enumerate(projs):
            loc = entity_locations_class(fname, cam_nr)
            Annotator(loc, proj, block=True)

    data = []
    for nr in range(3):  # TODO(Adriaan) how do we know the nr cams
        loc = entity_locations_class(fname, nr)
        l = loc.locations()
        data.append(l)

    return data


def triangle_geom(src_rad, det_rad, rotation=False, shift=False):
    geoms = []
    for src_a in [0, 2 / 3 * np.pi, 4 / 3 * np.pi]:
        det_a = src_a + np.pi  # opposing
        src = src_rad * np.array([np.cos(src_a), np.sin(src_a), 0])
        det = det_rad * np.array([np.cos(det_a), np.sin(det_a), 0])
        geom = xray.StaticGeometry(
            source=VectorParameter(src),
            detector=VectorParameter(det),
            roll=ScalarParameter(None),
            pitch=ScalarParameter(None),
            yaw=ScalarParameter(None),
        )
        geoms.append(geom)

    if rotation:
        # transform the whole geometry by a global rotation, this is the
        # same as if the phantom rotated

        rotation_roll = ScalarParameter(0.)
        rotation_pitch = ScalarParameter(0.)
        rotation_yaw = ScalarParameter(0.)

        for i in range(len(geoms)):
            geoms[i] = xray.transform(geoms[i],
                                      rotation_roll,
                                      rotation_pitch,
                                      rotation_yaw)

    if shift:
        shift_param = VectorParameter(np.array([0., 0., 0.]), optimize=True)
        for i in range(len(geoms)):
            geoms[i] = xray.shift(geoms[i], shift_param)

    return geoms


def astra_reco(reco, detector, geoms, t, algo='SIRT', voxels_x=400):
    # detector_mid = DETECTOR_ROWS // 2
    # offset = DETECTOR_ROWS // 2 - 0
    # sinogram = p[0, :, recon_height_range,
    #            recon_width_range.start:recon_width_range.stop]
    # recon_height_range = range(detector_mid - offset, detector_mid + offset)
    # recon_width_range = range(DETECTOR_COLS)
    # recon_height_length = int(len(recon_height_range))
    # recon_width_length = int(len(recon_width_range))
    p = reco.load_sinogram(t_range=range(t, t + 1), median=True)

    vectors = np.array([geom2astravec(g, detector) for g in geoms])
    proj_id, proj_geom = reco.sino_gpu_and_proj_geom(p, vectors)
    vol_id, vol_geom = reco.backward(proj_id, proj_geom, algo=algo,
                                     voxels_x=voxels_x)

    return vol_id, vol_geom


def astra_residual(reco, detector, vol_id, vol_geom, t, geoms=None):
    """Projects then substracts a volume with `vol_id` and `vol_geom`
    onto projections from `projs_path` with `nrs`.

    Using `geoms` or `angles`. If geoms are perfect, then the residual will be
    zero. Otherwise it will show some geometry forward-backward mismatch.
    """
    vectors = np.array([geom2astravec(g, detector) for g in geoms])
    sino_id, proj_geom = reco.sino_gpu_and_proj_geom(
        0.,  # zero-sinogram
        vectors
    )

    proj_id = reco.forward(
        volume_id=vol_id,
        volume_geom=vol_geom,
        projection_geom=proj_geom,
    )

    p = reco.load_sinogram(t_range=range(t, t + 1), median=True)
    return p - reco.sinogram(proj_id)


def plot_residual(res, vmin=None, vmax=None, title=None):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=1, ncols=3)
    if title is not None:
        plt.title(title)

    for i in range(3):
        im = axs[i].imshow(res[:, i], vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axs[i])

    plt.pause(.0001)
