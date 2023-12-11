# CaTE
CaTE (Calibration of Tomographic Equipment) is a small package for calibration of X-ray CT set-ups. It is essentially 
a utility wrapping [_scipy.optimize.least_squares_](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
It is written to find a geometry after scanning an object that features
marker points, such as metal balls.
The user provides a parametrized geometry, as well as annotated data of 
the projected markers. SciPy's NLLS solver subsequently looks for geometry parameters that
explain the markers.

CaTE is written with flexibility in mind. The parameters that may be optimized
are:
 - Geometries: tube locations, detector locations, as well
   as detector orientations (in roll-pitch-yaw format). Geometries can share
   parameters: for instance one source location can be used for multiple
   detectors.
 - Geometry transformations can be parametrized. For instance by rotations of
   the object and/or set-up.
 - Marker locations. Marker points in the object do not need to be known
   a-priori, but can be optimized together with the sought geometry.

Converts geometries back-and-forth to the format that is respected by the
ASTRA ToolBox.

## Method 
In X-ray tomography, the goal is to invert the relation Ax=y, i.e. to find
a volume x that explains radiographic projection data y. Here A is the linear
X-ray transform operator, encoding the physics.

Positions and orientations of the tubes, detectors, and object are usually
entered as parameters psi of the reconstruction algorithm (SIRT, FDK, ...), and 
parametrize
the forward operator, i.e. A=A_psi. Incorrect parameters lead to
blur and artifacts in the solution. Because jointly optimizing A_psi x=y over
psi and x may be inaccurate and computationally expensive, we find
psi using a calibration procedure.

The markerprocedure in its generality:
1. Place a markerobject in the scanning set-up. 
   1. The markerobject must be rigid.
   2. A markerobject has several points m_i, i=1...N. They can be anything,
   as long as they are **uniquely recognized** in most projection
   images. Ideas are: metal balls, points of needles. An object can thus be
   created easily and ad hoc.
   3. It is not necessary to know the (relative) positions of the
   markers. 
2. Rotate or move the markerobject and/or change scanning geometry. Collect
radiographic projections of the markerobject for several positions.
Number each geometry psi_j, j=1...T.
   1. All sorts of geometries are allowed. E.g., multiple sources/detectors, a moving
   detector/source, tiled scan.
3. For each marker point m_i, identify its location in the
projection image of geometry psi_j as p_i,j. This can be done using annotation, 
or in some cases using a tracking algorithm for markers (code not provided).
4. Initialize a guess for the geometry psi_1. Define the geometry of the later
projections as transformations of previous, i.e. psi_j = g(psi_1, ... psi_{t-1}).
   - If precise transformations are unknown, for instance because of
   uncertainty in rotation table/object/detector movement, auxiliary
   optimization variables phi can be introduced to encode rotation angle,
   speed of movement, etc. This leads to
   psi_j = psi_j(phi_1, ..., psi_{t-1}, phi).
6. Using an analytical X-ray projection operator B_psi, solve
min_psi,phi,m sum_j ||B_{psi_j} m - p_j||.

### Simplified example
The following example is simplified for the purpose of demonstration. Here
the geometry has a source position that requires calibration, but we trust that
we have a confident detector position. A
rotation table rotates the object, containing two markers, to three different
positions. We guess that the rotation angle is about pi/3, but we are not sure.
```python
# Suppose we don't know precisely where the source is, but we do know where
# the detector is.
source = VectorParameter(np.array([-10, 0, 0]))
detector = np.array([10, 0, 0])
geometry_1 = Geometry(source, detector, ...)

# Suppose there is a rotation table, and the geometry rotates clockwise, but
# we don't know how many degrees exactly.
angle = ScalarParameter(np.pi / 3)  # np.pi/10 is our guess
geometry_2 = rotate(geometry_1, yaw=angle)
geometry_3 = rotate(geometry_2, yaw=angle) # reuse parameter

# We want to optimize over the marker positions, since we don't know them.
# Therefore they are `VectorParameter`s.
markers = {'marker_1': VectorParameter([0.0, 0.2, 0.0]),
           'marker_2': VectorParameter([0.1, 0.3, -0.2]),
           ... }

# observed projections of the markers at each geometry
data = {...} 

# `XrayOptimizationProblem` is CaTE's helper for retrieving everything
# in a format that scipy likes.
problem = XrayOptimizationProblem(markers,
                                  [geometry_1, geometry_2, geometry_3],
                                   data)

# let's go:
scipy.optimize.least_squares(
    # uses `problem.__call__`:
    fun=problem,  
    # all marker points, geometry parameters, and auxiliary variables:
    x0=params2ndarray(problem.params()))

# solver has finished, parameters were updated in-place:
print(geometry_1.source)
print(markers['marker_1'])
```

## Documentation

### Geometry prescription
A`Geometry` is an encoding of a 
 - a `source`, that can be a `np.array` or `VectorParameter`;
 - a `detector`, that can be a `np.array` or `VectorParameter`;
 - the orientation of detector is encoded in two ways:
   - as a `roll`, `pitch` and `yaw` (w.r.t. world coordinates);
   - as an orthogonal pair of unit vectors describing the detector axis `u` and `v` (w.r.t. world coordinates).

A few notes:
 - `detector` is the central detector point;
 - There is no notion of "detector pixels". Optimization is in physical units.
   Conversion back and forth to pixel units can be done before and
   after the optimization. 

### Annotation tool
The source code includes a provisional annotation tool, with _Matplotlib_. To
use it the user must subclass `EntityLocations`, providing names for each marker
seen in the object. Please see _annotate.py_. The annotation tool can be
improved, please provide a PR if you do.
