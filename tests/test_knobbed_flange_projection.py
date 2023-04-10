import sys

import copy
import numpy as np
import open3d as o3d

sys.path.append(".")
import pyransac3d as pyrsc



flange = pyrsc.KnobbedFlange()

cloud = np.array([[0,-np.sqrt(2)/2,-np.sqrt(2)/2], [0,1,0], [0,0,1], [-.25, -1.1,0], [.3,.8,-.2]])
cloud += 1
cloud *= .1
o3dpoints = o3d.utility.Vector3dVector(cloud)
points = o3d.geometry.PointCloud(o3dpoints)
points.paint_uniform_color([1,.3,.3])

# o3d.visualization.draw_geometries([points])

data = flange._get_random_shape(cloud, 5, 8)
while(data is None):
    data = flange._get_random_shape(cloud, 5, 8)

mesh_cylinders = []
for cylinder in data: #[center, axis, radius, height]
    print(cylinder)
    print("-----------------------")
    center = cylinder[0]
    normal = cylinder[1]
    radius = cylinder[2]
    height = cylinder[3]

    R = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], normal)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.2)
    cen = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=center, size=0.5)
    mesh_rot = copy.deepcopy(mesh).rotate(R, center=[0, 0, 0])

    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    mesh_cylinder = mesh_cylinder.rotate(R, center=[0, 0, 0])
    mesh_cylinder = mesh_cylinder.translate((center[0], center[1], center[2]))

    mesh_cylinder.paint_uniform_color([.15,.15,.4])
    mesh_cylinders.append(mesh_cylinder);


o3d.visualization.draw_geometries(mesh_cylinders + [points])



#
