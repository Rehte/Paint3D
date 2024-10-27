import numpy as np
from collections import defaultdict

import torch
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from trimesh import Trimesh

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    AmbientLights,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturesUV
)

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

class XRayMesh:
    def __init__(
        self, 
        mesh, 
        cameras, 
        texture_size=1536, 
        channels=3, 
        device='cuda', 
        max_hits=2, 
        remove_backface_hits=True, 
        sampling_mode='nearest', 
        new_verts_uvs=None, 
        faces=None, 
        texture_init_maps=None
    ):
        self.mesh = mesh
        self.target_size = (texture_size,texture_size)
        self.channels = channels
        self.device = device
        self.max_hits = max_hits
        self.remove_backface_hits = remove_backface_hits
        self.sampling_mode = sampling_mode
        
        self.set_cameras(cameras)
        self.generate_occluded_geometry(faces, texture_init_maps, new_verts_uvs)
    
    def set_cameras(self, camera_poses, centers=None, scale=None):
        elev = torch.FloatTensor(camera_poses[0])
        azim = torch.FloatTensor(camera_poses[1])
        dist = torch.FloatTensor(camera_poses[2])
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=centers or ((0,0,0),))
        self.cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, scale_xyz=scale or ((1,1,1),))
        
    def generate_occluded_geometry(self, faces, texture_init_maps, new_verts_uvs):
        vertices = self.mesh.verts_packed().cpu().numpy()  # (V, 3) shape, move to CPU and convert to numpy
        # faces = self.mesh.faces_packed().cpu().numpy()  # (F, 3) shape, move to CPU and convert to numpy

        raycast = RaycastingImaging()

        self.visible_faces_list = []
        self.visible_texture_map_list = []
        self.mesh_face_indices_list = []
        
        for k, camera in enumerate(self.cameras):
            R = camera.R.cpu().numpy()
            T = camera.T.cpu().numpy()

            Rt = np.eye(4)  # Start with an identity matrix
            Rt[:3, :3] = np.swapaxes(R, 1, 2)  # Top-left 3x3 is the transposed rotation
            Rt[:3, 3] = T   # Top-right 3x1 is the inverted translation

            # mesh_frame = Trimesh(vertices=vertices, faces=self.mesh.faces_packed().cpu().numpy()).apply_transform(Rt)
            
            # self.mesh # Meshes -> Trimesh
            mesh_frame = Trimesh(vertices=self.mesh.verts_packed().cpu().numpy(), faces=self.mesh.faces_packed().cpu().numpy()).apply_transform(Rt)
            
            c2w = np.eye(4).astype(np.float32)[:3]
            raycast.prepare(image_height=512 * 3, image_width=512 * 3, c2w=c2w)
            ray_indexes, points, mesh_face_indices = raycast.get_image(mesh_frame, self.max_hits * 2 - 1)
            
            for i in range(self.max_hits):
                idx = i
                # idx = i * 2 if self.remove_backface_hits else i
                # print(f'Length of mesh faces indices {idx}', len(mesh_face_indices[idx]))
                # print(f"Mesh faces indices of {idx}", mesh_face_indices[idx])
                visible_faces = faces.verts_idx[mesh_face_indices[idx]]  # Only keep the visible faces
                self.mesh_face_indices_list.append(torch.tensor(mesh_face_indices[idx], dtype=torch.int64, device='cuda'))
                visible_faces = torch.tensor(visible_faces, dtype=torch.int64, device='cuda')
                
                # Initialize new mesh with Trimesh
                # Call init_mesh with the new mesh
                # Save the mesh info in a list

                self.visible_faces_list.append(visible_faces)
                
                # self.visible_texture_map_list.append(self.mesh.textures.faces_uvs_padded()[0, mesh_face_indices[idx]])
                self.visible_texture_map_list.append(faces.textures_idx[mesh_face_indices[idx]])
        
        new_map = torch.zeros(self.target_size+(self.channels,), device=self.device)
        expanded_texture_init_maps = texture_init_maps.repeat(len(self.cameras) * self.max_hits, 1, 1, 1)
        textures = TexturesUV(
            expanded_texture_init_maps, 
            self.visible_texture_map_list, 
            [new_verts_uvs] * len(self.cameras) * self.max_hits, # [self.mesh.textures.verts_uvs_padded()[0]] * len(self.cameras) * self.max_hits, 
            sampling_mode=self.sampling_mode
        )
        self.occ_mesh = Meshes(verts = [self.mesh.verts_packed()] * len(self.cameras) * self.max_hits, faces = self.visible_faces_list, textures = textures)
        self.occ_cameras = FoVOrthographicCameras(device=self.device, R=self.cameras.R.repeat_interleave(self.max_hits, 0), T=self.cameras.T.repeat_interleave(self.max_hits, 0), scale_xyz=self.cameras.scale_xyz.repeat_interleave(self.max_hits, 0))