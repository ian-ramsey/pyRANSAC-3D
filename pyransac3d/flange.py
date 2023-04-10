import random
import copy
import numpy as np

# from .aux_functions import rodrigues_rot


class Flange:
    """
    Implementation for flange RANSAC.

    This class finds the parameters (center, axis) defining a flange with given geometry.
    Currently this geometry is hardcoded, but could easily be made variable
    ---
    """

    FLANGE_GEOMETRY = {
        "h_top" : 0.325,
        "r_top" : .85,
        \
        "r_inner" : .45,
        \
        "h_base" : 0.45,
        "r_base" : 1
    }

    def __init__(self):
        self.inliers = []
        self.center = []
        self.axis = []
        self.radius = 1

    def _get_random_shape(self, pts, n_points):
        """
        Helper method that samples 3 points and creates 2 corresponding flanges,
        one for each orientation. The points are assumed to all lie on the largest
        disc of the flange.

        returns the component shapes
        - center, planes1, cylinders1, planes2, cylinders2

        on error, returns -1, -1, -1, -1, -1
        """
        # Samples 3 random points
        id_samples = random.sample(range(0, n_points), 3)
        pt_samples = pts[id_samples]

        # We have to find the plane equation described by those 3 points
        # We find first 2 vectors that are part of this plane
        # A = pt2 - pt1
        # B = pt3 - pt1

        vecA = pt_samples[1, :] - pt_samples[0, :]
        vecA_norm = vecA / np.linalg.norm(vecA)
        vecB = pt_samples[2, :] - pt_samples[0, :]
        vecB_norm = vecB / np.linalg.norm(vecB)

        # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
        vecC = np.cross(vecA_norm, vecB_norm)
        vecC = vecC / np.linalg.norm(vecC)

        # Compute center of 3 points
        # TODO: change this to radial center, not average.
        center = np.mean(pt_samples, axis = 0)
        for i in range(3): #check validity of points
            if (np.linalg.norm(center - pt_samples[i,:]) > 1):
                return -1,-1,-1,-1,-1 #error


        #compute both orientations
        planes1, cylinders1 = self._make_shape(center, vecC, pt_samples)
        planes2, cylinders2 = self._make_shape(center, -vecC, pt_samples)

        return center, planes1, cylinders1, planes2, cylinders2



    def _make_shape(self, center, axis, pt_samples):
        """
        create a single flange with given center and axis
        return 3 planes and 3 cylinders.

        plane:[axis0, axis1, axis2, k]

        cylinder: [axis, center, radius]
        """
        # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[2]*z = k
        # We have to use a point to find k
        k = np.sum(np.multiply(axis, pt_samples[1, :]))
        plane_eq = [axis[0], axis[1], axis[2], k]

        planes = [[], [], []]
        for i in range(3):
            planes[i] = copy.copy(plane_eq)

        planes[0][3] -= self.FLANGE_GEOMETRY["h_base"];
        planes[2][3] += self.FLANGE_GEOMETRY["h_top"];

        cylinders = [[axis, center - (self.FLANGE_GEOMETRY["h_base"] / 2) * axis, self.FLANGE_GEOMETRY["r_base"]],
                     [axis, center + (self.FLANGE_GEOMETRY["h_top"] / 2) * axis, self.FLANGE_GEOMETRY["r_top"]],
                     [axis, center + (self.FLANGE_GEOMETRY["h_top"] / 2) * axis                 \
                                    - (self.FLANGE_GEOMETRY["h_base"] / 2) * axis, self.FLANGE_GEOMETRY["r_inner"]] \
                    ]

        return planes, cylinders



    def _project_inliers(self, pts, planes, cylinders, thresh):
        """
        Given a flange defined by (planes, cylinders), project pts onto
        said flange and find inliers.

        Returns the inliers found
        """
        projection_errors = np.array([
            self._project_disc(pts, planes[0], cylinders[2], cylinders[0]),
            self._project_disc(pts, planes[1], cylinders[1], cylinders[0]),
            self._project_disc(pts, planes[2], cylinders[2], cylinders[1]),
            self._project_cylinder(pts, cylinders[0], planes[0], planes[1]),
            self._project_cylinder(pts, cylinders[1], planes[1], planes[2]),
            self._project_cylinder(pts, cylinders[2], planes[0], planes[2]) \
        ])

        dist_pt = np.minimum.reduce(projection_errors, axis = 0)

        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]

        return pt_id_inliers

    def _project_disc(self, points, plane, cylinder_inner, cylinder_outer):
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

        return np.linalg.norm(proj - points, axis = 1)

    def _project_cylinder(self, points, cylinder, plane_bottom, plane_top):
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

        return np.linalg.norm(proj - points, axis = 1)




    def fit(self, pts, radius = 1, thresh=0.2, maxIteration=10000):
        """
        Find the parameters (center, axis) defining a flange.
        Notably, radius must be given in advance.

        :param pts: 3D point cloud as a numpy array (N,3).
        :param radius: the radius of the flange. if radius == -1, assumes variable radius #TODO, implement variable radius
        :param thresh: Threshold distance from the cylinder hull which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.

        :returns:
        - `center`: Center of the flange np.array(1,3) which the flange axis is passing through.
        - `axis`: Vector describing flange's axis np.array(1,3).
        - `inliers`: Inlier's index from the original point cloud.
        ---
        """

        self.radius = radius;
        for key in self.FLANGE_GEOMETRY.keys():
            self.FLANGE_GEOMETRY[key] *= radius

        n_points = pts.shape[0]
        best_inliers = []

        print('fitting a flange...')
        print('\n', end = '')
        for it in range(maxIteration):
            if it % (maxIteration / 100) == 0:
                print("\033[F" , end = '')
                print(f"{(it*100)//maxIteration}% complete!")

            data = self._get_random_shape(pts, n_points)
            center = data[0]
            shape = data[1:]

            if shape[0] == -1: #error
                continue;   #TODO: choose more intelligent sampling method.

            for i in range(2):
                planes = shape[2*i]
                cylinders = shape[2*i + 1]

                pt_id_inliers = self._project_inliers(pts, planes, cylinders, thresh)

                if len(pt_id_inliers) >= len(best_inliers):
                    best_inliers = pt_id_inliers
                    self.inliers = best_inliers
                    self.center = center
                    self.axis = planes[1][0:3]

        return self.center, self.axis, self.inliers
















#
