import copy

import numpy as np
from fbrct.astra_reco import Reconstruction

from cate import xray
from delft.util import (triangle_geom)
from needles import util as needles_util
from numerical_xray import rotate_markers

SOURCE_RADIUS = 86.5
DETECTOR_RADIUS = 13.5
DETECTOR_ROWS = 1548  # including ROI
DETECTOR_COLS = 400  # including ROI
DETECTOR_COLS_SPEC = 1524  # also those outside ROI
DETECTOR_WIDTH_SPEC = 30.2  # cm, also outside ROI
DETECTOR_HEIGHT = 30.7  # cm, also outside ROI
DETECTOR_WIDTH = DETECTOR_WIDTH_SPEC / DETECTOR_COLS_SPEC * DETECTOR_COLS  # cm
FRAMERATE = 90  # seconds
DETECTOR_PIXEL_WIDTH = DETECTOR_WIDTH / DETECTOR_COLS
DETECTOR_PIXEL_HEIGHT = DETECTOR_HEIGHT / DETECTOR_ROWS
APPROX_VOXEL_WIDTH = DETECTOR_PIXEL_WIDTH / (
    SOURCE_RADIUS + DETECTOR_RADIUS) * SOURCE_RADIUS
APPROX_VOXEL_HEIGHT = DETECTOR_PIXEL_HEIGHT / (
    SOURCE_RADIUS + DETECTOR_RADIUS) * SOURCE_RADIUS

detector_dict = {'rows': DETECTOR_ROWS,
                 'cols': DETECTOR_COLS,
                 'pixel_width': DETECTOR_PIXEL_WIDTH,
                 'pixel_height': DETECTOR_PIXEL_HEIGHT}
detector = xray.Detector(detector_dict['rows'], detector_dict['cols'],
                         detector_dict['pixel_width'],
                         detector_dict['pixel_height'])

data_dir = "/bigstore/adriaan/data/evert/2020-02-19 3D paper dataset 1/2020-02-12"
projs_path = f'{data_dir}/pre_proc_20mm_ball_62mmsec_03'
fulls_path = f'{data_dir}/pre_proc_Full_11_6lmin'

reco = Reconstruction(projs_path,
                      fulls_path,
                      detector_dict,
                      expected_voxel_size_x=APPROX_VOXEL_WIDTH,
                      expected_voxel_size_z=APPROX_VOXEL_HEIGHT)
geoms = triangle_geom(SOURCE_RADIUS, DETECTOR_RADIUS,
                      rotation=True, shift=True)

vmin, vmax = -0.02, 0.02

# t = 450
# vol_id, vol_geom = astra_reco(reco, detector, geoms, t)
# x = reco.volume(vol_id)
# plot_residual(astra_residual(reco, detector, vol_id, vol_geom, t, geoms),
#               'before')
# astra.clear()
# pq.image(x)
# projs = _load_median_projs(projs_path, fulls_path, t)
# for i in range(3):
#     plt.figure()
#     plt.imshow(projs[i])
#     plt.pause(.1)

markers = np.load('../needles/markers_11dec2020_calibrated_14jan2021.npy',
                  allow_pickle=True).item()  # type: dict
for m in markers.values():
    m.value /= 10.  # convert mm to cm

# data = annotated_data(projs_path,
#                       fulls_path,
#                       300,  # any frame will probably do
#                       DelftNeedleEntityLocations,
#                       'needles_in_delft_xxjan2021.npy',
#                       open_annotator=True)

fake_geoms = np.load('geoms_metal_looking_pieces_14jan2021.npy',
                     allow_pickle=True)


def fake_data(markers: dict, r, p, y, shift):
    rotate_markers(markers, r, p, y)

    markers = copy.deepcopy(markers)
    for m in markers.values():
        m.value += shift

    projs = xray.xray_multigeom_project(fake_geoms, markers)
    return projs


shift = np.ones(3) * 0.
r, p, y = 0 * np.pi / 3, 0 * np.pi / 6, np.pi / 6
r, p, y = 0, 0, 0
data = fake_data(markers, r=r, p=p, y=y, shift=shift)

# TODO: change data into real data

for d1, d2 in zip(data, xray.xray_multigeom_project(geoms, markers)):
    xray.plot_projected_markers(d1, d2, det=detector, det_padding=1.2)

needles_util.run_calibration(geoms, markers, data, verbose=False)

# vol_id, vol_geom = astra_reco(reco, detector, geoms, t)
# x = reco.volume(vol_id)
# pq.image(x)
# res = astra_residual(reco, detector, vol_id, vol_geom, t, geoms)
# plot_residual(res, 'after')

# vectors = np.array([geom2astravec(g, detector) for g in geoms])
# np.save('geoms_new.npy', vectors)  # TODO date

for d1, d2 in zip(data, xray.xray_multigeom_project(geoms, markers)):
    xray.plot_projected_markers(d1, d2, det=detector)

import matplotlib.pyplot as plt
plt.figure()
plt.show()