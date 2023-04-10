import sys

import copy
import numpy as np
import open3d as o3d

sys.path.append(".")
import pyransac3d as pyrsc



flange = pyrsc.Flange()


for i in range(5):
    pcd = o3d.io.read_point_cloud(f"tests/dataset/flange_{1+i}.txt", format = 'xyz')
    print(pcd)
    # o3d.visualization.draw_geometries([pcd])

    points = np.asarray(pcd.points)
    radius = .08
    center, axis, inliers = flange.fit(points, radius = radius, thresh = 0.003, maxIteration = 5000)
    axis = np.array(axis)

    # print(center)
    # print(axis)
    # print(inliers)

    mesh_cylinders = [0,0,0]
    colors = [[.05,.15,.15], [.02,.2,.2], [.8,.8,.8]]
    heights = [.45, .325, .775+.02]
    heights = radius * np.array(heights)
    radii = [1, .85, .45]
    radii = radius * np.array(radii)
    centers = [center - (heights[0] / 2) * axis,
               center + (heights[1] / 2) * axis,
               center + (heights[1] / 2) * axis  \
                        - (heights[0] / 2) * axis]

    for i in range(3):
        R = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], axis)
        if (i == 0):
            R = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], -axis)

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.2)
        cen = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=centers[i], size=0.5)
        mesh_rot = copy.deepcopy(mesh).rotate(R, center=[0, 0, 0])

        mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radii[i], height=heights[i])
        mesh_cylinder.compute_vertex_normals()
        mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
        mesh_cylinder = mesh_cylinder.rotate(R, center=[0, 0, 0])
        mesh_cylinder = mesh_cylinder.translate((centers[i][0], centers[i][1], centers[i][2]))

        mesh_cylinder.paint_uniform_color(colors[i])
        # o3d.visualization.draw_geometries([mesh_cylinder, points])
        mesh_cylinders[i] = mesh_cylinder

    inlier_points = pcd.select_by_index(inliers)
    inlier_points.paint_uniform_color([1,0,0])

    not_inlier_points = pcd.select_by_index(inliers, invert = True)
    not_inlier_points.paint_uniform_color([0,1,1])

    o3d.visualization.draw_geometries(mesh_cylinders + [inlier_points, not_inlier_points])
