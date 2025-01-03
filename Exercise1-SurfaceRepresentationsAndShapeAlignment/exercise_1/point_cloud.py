"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """
    # ---------------------------------------------------------------------------------------------------------------
    triangle_vertices = np.dstack([vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]])

    # areas computed using corss-product of the vertex values
    triangle_areas = 0.5 * np.linalg.norm(np.cross(triangle_vertices[:, 1, :] - triangle_vertices[:, 0, :],
                                                   triangle_vertices[:, 2, :] - triangle_vertices[:, 0, :]), axis=1)
    
    # since areas are positive
    triangle_probabilities = triangle_areas / triangle_areas.sum()

    # random choice (p argument ensures choosing based on waiting as opposed to uniformly) for n samples
    chosen_triangles = np.random.choice(range(triangle_areas.shape[0]), size=n_points, p=triangle_probabilities)
    
    # to get vertex values from chosen triangles
    chosen_vertices = triangle_vertices[chosen_triangles, :, :]

    # barycentric interpolation for sampling points
    r1 = np.random.rand(n_points, 1)
    r2 = np.random.rand(n_points, 1)
    u = (1 - np.sqrt(r1))
    v = np.sqrt(r1) * (1 - r2)
    w = np.sqrt(r1) * r2

    # get the coordinates of sampled point and return
    result_xyz = (u * chosen_vertices[:, :, 0]) + (v * chosen_vertices[:, :, 1]) + (w * chosen_vertices[:, :, 2])
    return result_xyz
    # ---------------------------------------------------------------------------------------------------------------
