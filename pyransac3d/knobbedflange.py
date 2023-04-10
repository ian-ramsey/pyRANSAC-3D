import random
import copy
import numpy as np

# from .aux_functions import rodrigues_rot


class KnobbedFlange:
    """
    Implementation for flange with knobs RANSAC.

    This class finds the parameters (center, axis, height) defining a flange with given geometry.
    Currently this geometry is hardcoded, but could easily be made variable
    ---
    """

    # FLANGE_GEOMETRY = {
    #     "h_top" : 0.325,
    #     "r_top" : .85,
    #     \
    #     "r_inner" : .45,
    #     \
    #     "h_base" : 0.45,
    #     "r_base" : 1
    # }

    #restricts height of flange to within this multiplier of radius
    HEIGHT_UPPER_MULTPLIER = .6 #note this applies both above and below
    #restricts height of flange to within this multiplier of radius
    HEIGHT_LOWER_MULTPLIER = .1 #note this applies both above and below

    #restricts knob point to outiside this multiplier of radius
    KNOB_INNER_MULTIPLIER = 0.5
    #restricts knob point to inside this multiplier of radius
    KNOB_OUTER_MULTIPLIER = 0.85

    #fancy statistics: r* = (m-c*(.14))/(1-.14)
    EXPECTED_MIN_VALUE = .14

    def __init__(self):
        self.inliers = []
        self.cylinders = []

    def _get_random_shape(self, pts, n_points, knob_count):
        """
        Helper method that samples 3 points and creates 2 corresponding flanges,
        one for each orientation. The points are assumed to all lie on the largest
        disc of the flange.

        returns the component cylinders
        - center, [cy1, cy2, ...]

        where cyi = [center, axis, radius, height]


        on error, returns None
        """
        # Samples 5 random points
        # id_samples = [0,1,2,3,4] #for testing
        id_samples = random.sample(range(0, n_points), 5)
        pt_samples = pts[id_samples]

        ##########################################
        # Initial orientation calculation
        ##########################################

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

        # Compute centers of the sample
        center = np.mean(pt_samples, axis = 0)
        center = center - np.dot(vecC, center - pt_samples[0, :])*vecC #project onto plane

        #construct central plane
        # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[2]*z = k
        # We have to use a point to find k
        axis = vecC
        k = np.sum(np.multiply(axis, pt_samples[1, :]))
        central_plane_eq = [axis[0], axis[1], axis[2], k]

        ##########################################
        # Validate remaining points
        ##########################################
        #decide upper and lower points i.e. orientation
        #compute distance from central axis
        d3 =  np.linalg.norm(np.cross(axis, (pt_samples[3,:] - center)))
        d4 =  np.linalg.norm(np.cross(axis, (pt_samples[4,:] - center)))

        bottom_point = None
        top_point = None
        radius = None
        knob_center_r = None
        if (d3 < d4):
            bottom_point = pt_samples[4,:]
            top_point = pt_samples[3,:]
            radius = d4
            knob_center_r = d3
        else:
            bottom_point = pt_samples[3,:]
            top_point = pt_samples[4,:]
            radius = d3
            knob_center_r = d4

        # print("---------------------")
        # print(pt_samples)

        #validate that all points are inside radius
        D = np.linalg.norm(np.cross(axis, (pt_samples - center)), axis = 1)
        if (radius + 0.00001 < np.max(D)): #bottom point not outermost
            # print("----------")
            # print(radius)
            # print(np.max(D))
            # print("1")
            return None

        if (knob_center_r < radius*self.KNOB_INNER_MULTIPLIER)\
            or (knob_center_r > radius*self.KNOB_OUTER_MULTIPLIER): #bad knob radius
            # print("2")
            # print(radius*self.KNOB_INNER_MULTIPLIER)
            # print(radius*self.KNOB_OUTER_MULTIPLIER)
            # print()
            # print(knob_center_r)
            # print("\n")
            return None


        #check that signed distance of 4th and 5th points from plane are opposite
        d_bottom = (np.dot(bottom_point,axis) - central_plane_eq[3]) #0 indexed
        d_top = (np.dot(top_point,axis) - central_plane_eq[3])
        if (d_bottom*d_top) >= 0: #same direction. reject sample
            # print("3")
            return None

        axis = np.sign(d_top)*axis

        knob_height = abs(d_top)
        base_height = abs(2*d_bottom)

        #validate height
        if (knob_height > radius * self.HEIGHT_UPPER_MULTPLIER) \
            or (knob_height < radius * self.HEIGHT_LOWER_MULTPLIER):
            # print("4")
            # print("\n---------------------")
            # print(pt_samples)
            # print()
            # print(radius * self.HEIGHT_UPPER_MULTPLIER)
            # print(radius * self.HEIGHT_LOWER_MULTPLIER)
            # print()
            # print(knob_height)
            # print("---------------------")

            return None

        if (base_height > radius * self.HEIGHT_UPPER_MULTPLIER) \
            or (base_height < radius * self.HEIGHT_LOWER_MULTPLIER):
            # print("5")
            # print()
            # print(radius * self.HEIGHT_UPPER_MULTPLIER)
            # print(radius * self.HEIGHT_LOWER_MULTPLIER)
            # print()
            # print(base_height)
            # print()
            # print(radius)
            return None

        ####################################
        #estimate knob radius
        ####################################

        #estimate knob radius
        big_theta = 2*np.pi/knob_count
        small_theta = big_theta/2
        rot_center = center + np.dot(axis, top_point - center)*axis
        knob_center = lambda x : rot_center + (np.cos(x*big_theta)*(top_point - rot_center) \
                                            +np.sin(x*big_theta)*np.cross(axis, (top_point - rot_center)))
        knob_centers = [knob_center(knob_idx) for knob_idx in range(knob_count)]
        # print()
        # print(rot_center)
        # print(knob_centers)

        distances = [np.min([np.linalg.norm(point-center+rot_center - kcenter) for kcenter in knob_centers])
                            for point in pt_samples[0:3,:]]
        # print(distances)
        sample_min = np.min(distances)
        knob_radius = (sample_min-knob_center_r*self.EXPECTED_MIN_VALUE)\
                        /(1 - self.EXPECTED_MIN_VALUE)


        #validate knob_radius
        if (knob_radius + knob_center_r > radius) \
            or (knob_radius > knob_center_r*np.sin(small_theta)):
            # print(f"knob too wide!")
            # print(f"\tradius: {radius}")
            # print(f"\tknob_radius: {knob_radius}")
            # print(f"\trmax: {knob_center_r*np.sin(small_theta)}")
            # print()
            # print(f"\tcenter: {center}")
            # print(f"\trot_center: {rot_center}")
            # print(f"\taxis: {axis}")
            return None


        ####################################
        #validations passed! Create shape
        ####################################

        # print()
        # print("success!")
        # print("New Geometry:")
        # print(f"\tcenter:{center}")
        # print(f"\taxis:{axis}")
        # print(f"\tradius:{radius}")
        # print(f"\th_top:{knob_height}")
        # print(f"\th_base:{base_height}")
        # print(f"\tsample_min:{sample_min}")
        # print(f"\tknob_center_radius:{knob_center_r}")
        # print(f"\tknob_radius:{knob_radius}")

        base_cylinder = (center - base_height/2*axis, axis, radius, base_height)
        knob_cylinders = [(kcenter - knob_height/2*axis, axis, knob_radius, knob_height) \
                                for kcenter in knob_centers]
        cylinders = [base_cylinder] + knob_cylinders




        return cylinders



    def _project_inliers(self, pts, cylinders, thresh):
        """
        Given a flange defined by (planes, cylinders), project pts onto
        said flange and find inliers.

        Returns the inliers found
        """
        projection_errors = []
        for cylinder in cylinders:
            center = cylinder[0]
            axis = cylinder[1]
            radius = cylinder[2]
            height = cylinder[3]

            projection_errors += [
                self._project_disc(pts, axis, center + height/2*axis, radius),
                self._project_cylinder(pts, axis, center, radius, height),
                self._project_disc(pts, axis, center - height/2*axis, radius)
            ]

        dist_pt = np.minimum.reduce(projection_errors, axis = 0)

        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]

        return pt_id_inliers

    def _project_disc(self, points, axis, center, radius):
        D =  np.linalg.norm(np.cross(axis, (points - center)), axis = 1)  #distance to inner line

        proj = np.array(points)

        outer_mask = D > radius #outside outer radius; project in
        proj[outer_mask] += ((D[outer_mask]- radius)/D[outer_mask])[:,None]   \
                            * ((center-proj[outer_mask]) - np.outer(np.dot((center-proj[outer_mask]),axis) , axis))

        #can now assume proj is inside flange bounding donut
        k = np.dot(center, axis)
        proj = proj - np.outer((np.dot(proj,axis) - k), axis)

        return np.linalg.norm(proj - points, axis = 1)

    def _project_cylinder(self, points, axis, center, radius, height):
        plane_top = [axis[0], axis[1], axis[2], np.dot(center,axis) + height/2]
        plane_bottom = [axis[0], axis[1], axis[2], np.dot(center,axis) - height/2]

        D = np.dot(points, axis) #distance to origin along points

        proj = np.array(points)

        above_mask = D > plane_top[3] #above flange; project inside
        proj[above_mask] = proj[above_mask] - np.outer((np.dot(proj[above_mask],axis) - plane_top[3]), axis)

        below_mask = D < plane_bottom[3] #below flange; project inside
        proj[below_mask] = proj[below_mask] - np.outer((np.dot(proj[below_mask],axis) - plane_bottom[3]), axis)

        #can now assume proj is inside vertical flange region.
        D =  np.linalg.norm(np.cross(axis, (points - center)), axis = 1)  #distance to inner line
        proj += ((D-radius)/D)[:,None]   \
                            * ((center-proj) - np.outer(np.dot((center-proj),axis) , axis))

        return np.linalg.norm(proj - points, axis = 1)




    def fit(self, pts, knob_count, thresh=0.2, maxIteration=10000):
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

        n_points = pts.shape[0]
        best_inliers = []

        print('fitting a flange...')
        print('\n', end = '')
        for it in range(maxIteration):
            if it % (maxIteration / 100) == 0:
                print("\033[F" , end = '')
                print(f"{(it*100)//maxIteration}% complete!")

            cylinders = self._get_random_shape(pts, n_points, knob_count)

            if cylinders == None: #error
                continue;

            pt_id_inliers = self._project_inliers(pts, cylinders, thresh)

            if len(pt_id_inliers) >= len(best_inliers):
                best_inliers = pt_id_inliers
                self.inliers = best_inliers
                self.cylinders = cylinders

        return self.cylinders, self.inliers
















#
