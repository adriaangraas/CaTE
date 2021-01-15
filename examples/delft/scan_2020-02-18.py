import astra
import matplotlib.pyplot as plt
import pyqtgraph as pq
from fbrct.astra_reco import Reconstruction

from cate import annotate, xray
from delft.util import (_load_median_projs, triangle_geom, astra_reco, plot_residual,
    astra_residual, annotated_data,)
from needles.util import (pixels2coords, run_initial_marker_optimization,
    geom2astravec)
import numpy as np


class WeirdMetalLookingPieces(annotate.EntityLocations):
    ENTITIES = (
        'Biggest metal spot',
        'Just below ball',  # at t =~ 250 or 300 I don't remember
        'Far below ball',  # at t =~ 250
    )

    @staticmethod
    def nr_entities():
        return len(WeirdMetalLookingPieces.ENTITIES)

    @staticmethod
    def get_iter():
        return iter(WeirdMetalLookingPieces.ENTITIES)


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
geoms = triangle_geom(SOURCE_RADIUS, DETECTOR_RADIUS)

vmin, vmax = -0.02, 0.02

# t = 450
# vol_id, vol_geom = astra_reco(reco, detector, geoms, t)
# x = reco.volume(vol_id)
# plot_residual(astra_residual(reco, detector, vol_id, vol_geom, t, geoms),
#               vmin=vmin, vmax=vmax, title='before')
# astra.clear()
# pq.image(x)

# projs = _load_median_projs(projs_path, fulls_path, 450)
# for i in range(3):
#     plt.figure()
#     plt.imshow(projs[i])
#     plt.pause(.1)

data = annotated_data(projs_path,
                      fulls_path,
                      300,  # any frame will probably do
                      WeirdMetalLookingPieces,
                      'examples/delft/metal_looking_pieces.npy',
                      open_annotator=False)
pixels2coords(data, detector)

# markers = xray.markers_from_leastsquares_intersection(
#     geoms, data,
#     optimizable=False,
#     plot=False)
markers = run_initial_marker_optimization(
    geoms,
    data,
    nr_iters=3,
    plot=False)

items = zip(data, xray.xray_multigeom_project(geoms, markers))
for i, j in items:
    xray.plot_projected_markers(i, j, det=detector)


t = 450
vol_id, vol_geom = astra_reco(reco, detector, geoms, t)
x = reco.volume(vol_id)
# astra.clear()
pq.image(x)
res = astra_residual(reco, detector, vol_id, vol_geom, t, geoms)
plot_residual(res, vmin=vmin, vmax=vmax, title='after')

vectors = np.array([geom2astravec(g, detector) for g in geoms])
np.save('examples/delft/vectors_metal_looking_pieces_14jan2021.npy', vectors)
np.save('examples/delft/geoms_metal_looking_pieces_14jan2021.npy', geoms)

for d1, d2 in zip(data, xray.xray_multigeom_project(geoms, markers)):
    xray.plot_projected_markers(d1, d2, det=detector)
