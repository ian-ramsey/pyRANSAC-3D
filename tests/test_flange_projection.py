import sys

import copy
import numpy as np
import open3d as o3d

sys.path.append(".")
import pyransac3d as pyrsc



flange = pyrsc.Flange()

cloud = np.array([[1,0,0], [0,1,0], [0,0,1]])
data = flange._get_random_shape(cloud, 3) #DO NOT DO THIS. I AM CALLING AN INTERNAL METHOD FOR TESTING PURPOSES.
center = data[0]
shape = data[1:]

for i in range(2):
    print()
    print('------------------------------')
    print()
    print(f"planes {i}:")
    for j in range(3):
        print(f"\t {shape[2*i][j]}")
    print()
    print(f"cylinders {i}:")
    for j in range(3):
        print(f"\t {shape[2*i+1][j]}")
    print()
print()
print('------------------------------')
print()
o3dpoints = o3d.utility.Vector3dVector(cloud)
points = o3d.geometry.PointCloud(o3dpoints)
points.paint_uniform_color([1,.3,.3])
# o3d.visualization.draw_geometries([points])

mesh_cylinders = [0,0,0]
colors = [[.05,.15,.15], [.02,.2,.2], [.8,.8,.8]]
heights = [.2, .1, .3+.001]
radii = [1, .7, .6]

for i in range(3):
    normal = shape[1][i][0]
    center = shape[1][i][1]


    R = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], normal)
    if (i == 0):
        R = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], -normal)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.2)
    cen = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=center, size=0.5)
    mesh_rot = copy.deepcopy(mesh).rotate(R, center=[0, 0, 0])

    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radii[i], height=heights[i])
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    mesh_cylinder = mesh_cylinder.rotate(R, center=[0, 0, 0])
    mesh_cylinder = mesh_cylinder.translate((center[0], center[1], center[2]))

    mesh_cylinder.paint_uniform_color(colors[i])
    # o3d.visualization.draw_geometries([mesh_cylinder, points])
    mesh_cylinders[i] = mesh_cylinder


o3d.visualization.draw_geometries(mesh_cylinders + [points])





cloud = []
num_points = 99
p0 = np.array([-1.5,-1,1])
p1 = np.array([1.5,2,0])
for i in range(num_points+1):
    point = p0 + i/num_points * (p1 - p0)
    point += 1/4*np.array([np.cos(4*np.pi*i/num_points), np.sin(4*np.pi*i/num_points), 2*(np.cos(2*np.pi*i/num_points) + np.sin(2*np.pi*i/num_points))])
    cloud.append(point)

disc_sets = [
    [0,2,0],
    [1,1,0],
    [2,2,1]
]

cylinder_sets = [
    [0, 0, 1],
    [1, 0, 2],
    [2, 1, 2]
]

projections = []
best_projections = []
points = np.array(cloud)

#copied code to access internal state; must copy again if internal code changes
for idx_set in disc_sets:
    plane = shape[0][idx_set[0]]
    cylinder_inner = shape[1][idx_set[1]]
    cylinder_outer = shape[1][idx_set[2]]

    axis = cylinder_inner[0]
    center = cylinder_inner[1]
    inner_radius = cylinder_inner[2]
    outer_radius = cylinder_outer[2]

    D =  np.linalg.norm(np.cross(axis, (points - center)), axis = 1)  #distance to inner line

    proj = np.array(points)

    inner_mask = D < inner_radius #inside inner tube; project out
    proj[inner_mask] += ((D[inner_mask]-inner_radius)/D[inner_mask])[:,None]  \
                        * ((center-proj[inner_mask]) - np.outer(np.dot((center-proj[inner_mask]),axis) , axis))

    outer_mask = D > outer_radius #outside outer radius; project in
    proj[outer_mask] += ((D[outer_mask]-outer_radius)/D[outer_mask])[:,None]   \
                        * ((center-proj[outer_mask]) - np.outer(np.dot((center-proj[outer_mask]),axis) , axis))

    #can now assume proj is inside flange bounding donut
    proj = proj - np.outer((np.dot(proj,axis) - plane[3]), axis)

    projections.extend(proj)

for idx_set in cylinder_sets:
    cylinder = shape[1][idx_set[0]]
    plane_bottom = shape[0][idx_set[1]]
    plane_top = shape[0][idx_set[2]]

    axis = cylinder[0]

    D = np.dot(points, axis)

    proj = np.array(points)

    above_mask = D > plane_top[3] #above flange; project inside
    proj[above_mask] = proj[above_mask] - np.outer((np.dot(proj[above_mask],axis) - plane_top[3]), axis)

    below_mask = D < plane_bottom[3] #below flange; project inside
    proj[below_mask] = proj[below_mask] - np.outer((np.dot(proj[below_mask],axis) - plane_bottom[3]), axis)

    #can now assume proj is inside vertical flange region.
    D =  np.linalg.norm(np.cross(axis, (points - cylinder[1])), axis = 1)  #distance to inner line
    proj += ((D-cylinder[2])/D)[:,None]   \
                        * ((cylinder[1]-proj) - np.outer(np.dot((cylinder[1]-proj),axis) , axis))

    projections.extend(proj)


projections = np.array(projections)

o3dpoints1 = o3d.utility.Vector3dVector(cloud)
points1 = o3d.geometry.PointCloud(o3dpoints1)
points1.paint_uniform_color([1,.5,.3])

o3dpoints2 = o3d.utility.Vector3dVector(projections)
points2 = o3d.geometry.PointCloud(o3dpoints2)
points2.paint_uniform_color([.8,.8,.5])

# mesh_planes = []
# for i in range(3):
#     plane = shape[0][i]
#     xy = [[-.5,-.5], [0,1.5], [1.5,0]]
#     xyz = [[point[0], point[1], 1/plane[2] * (plane[3] - plane[0]*point[0] - plane[1]*point[1])] for point in xy]
#     triangle = np.array([[0, 2, 1]]).astype(np.int32)
#
#     vertices = o3d.utility.Vector3dVector(np.array(xyz))
#     triangles = o3d.utility.Vector3iVector(triangle)
#     mesh_planes.append(o3d.geometry.TriangleMesh(vertices, triangles))

o3d.visualization.draw_geometries([points1, points2, points3] + mesh_cylinders)







#
