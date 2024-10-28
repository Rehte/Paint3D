import numpy as np
from collections import defaultdict

import torch
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from trimesh import Trimesh

def get_rays(directions, c2w, near = 1):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    # The origin of all rays is the camera origin in world coordinate

    H, W, _ = directions.shape
    # Generate a grid of pixel coordinates in camera space
    # These represent the ray origins in orthographic projection
    i, j = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))  # Normalized pixel grid
    
    # Assuming the camera is facing along the -z direction in camera space
    # The z values are fixed (e.g., at 'near' distance from the camera plane)
    pixel_positions_camera = np.stack([i, j, -np.ones_like(i) * near], axis=-1)  # (H, W, 3)
    
    # Transform pixel positions from camera space to world space using the c2w matrix
    pixel_positions_world = pixel_positions_camera @ c2w[:3, :3].T + c2w[:3, 3]  # (H, W, 3)
    
    # The ray origins are the pixel positions in world space
    rays_o = pixel_positions_world

    rays_o = pixel_positions_camera

    return rays_o, rays_d

def generate_rays(image_resolution, c2w):
    if isinstance(image_resolution, tuple):
        assert len(image_resolution) == 2
    else:
        image_resolution = (image_resolution, image_resolution)
    image_width, image_height = image_resolution

    # generate an array of screen coordinates for the rays
    # (rays are placed at locations [i, j] in the image)
    rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(
        2, image_height * image_width).T  # [h, w, 2]

    grid = rays_screen_coords.reshape(image_height, image_width, 2)
    
    i, j = grid[..., 1], grid[..., 0]
    directions = np.stack([np.zeros_like(i), np.zeros_like(i), np.ones_like(i)], -1) # (H, W, 3)

    rays_origins, ray_directions = get_rays(directions, c2w)
    rays_origins = rays_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    
    return rays_screen_coords, rays_origins, ray_directions


def ray_cast_mesh(mesh, rays_origins, ray_directions):
    intersector = RayMeshIntersector(mesh)
    index_triangles, index_ray, point_cloud = intersector.intersects_id(
        ray_origins=rays_origins,
        ray_directions=ray_directions,
        multiple_hits=True,
        return_locations=True)
    return index_triangles, index_ray, point_cloud

class RaycastingImaging:
    def __init__(self):
        self.rays_screen_coords, self.rays_origins, self.rays_directions = None, None, None

    def __del__(self):
        del self.rays_screen_coords
        del self.rays_origins
        del self.rays_directions

    def prepare(self, image_height, image_width, c2w=None):
        # scanning radius is determined from the mesh extent
        self.rays_screen_coords, self.rays_origins, self.rays_directions = generate_rays((image_height, image_width), c2w)
    
    def get_image(self, mesh, max_hits = 4):  #, features):
        # get a point cloud with corresponding indexes
        mesh_face_indexes, ray_indexes, points = ray_cast_mesh(mesh, self.rays_origins, self.rays_directions)

        ray_face_indexes = defaultdict(list)
        for ray_index, ray_face_index in zip(ray_indexes, mesh_face_indexes):
            ray_face_indexes[ray_index].append(ray_face_index)
            
        mesh_face_indices = [[] for _ in range(max_hits)]
        for i in range(max_hits):
            for ray_index, ray_face_index in ray_face_indexes.items():
                if i < len(ray_face_index):
                    mesh_face_indices[i].append(ray_face_index[i])

        mesh_face_indices = [np.unique(indexes) for indexes in mesh_face_indices]
        # print([mesh_face_indices[i].shape for i in range(max_hits)])
        return ray_indexes, points, mesh_face_indices
