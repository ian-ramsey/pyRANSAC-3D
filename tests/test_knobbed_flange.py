import sys

import copy
import numpy as np
import open3d as o3d

sys.path.append(".")
import pyransac3d as pyrsc



flange = pyrsc.KnobbedFlange()


for i in range(5):
    pcd = o3d.io.read_point_cloud(f"tests/dataset/flange_{1+i}.txt", format = 'xyz')
    print(pcd)
    o3d.visualization.draw_geometries([pcd])

    points = np.asarray(pcd.points)
    cylinders, inliers = flange.fit(pts = points, knob_count = 8, thresh = 0.003, maxIteration = 100000)

    # print(center)
    # print(axis)
    # print(inliers)
    print("Done!")
    mesh_cylinders = []
    for cylinder in cylinders: #[center, axis, radius, height]
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


    inlier_points = pcd.select_by_index(inliers)
    inlier_points.paint_uniform_color([1,0,0])

    not_inlier_points = pcd.select_by_index(inliers, invert = True)
    not_inlier_points.paint_uniform_color([0,1,1])

    o3d.visualization.draw_geometries(mesh_cylinders + [inlier_points, not_inlier_points])
