import os
import cv2
import numpy as np
import kaolin as kal
from PIL import Image
from loguru import logger
from pathlib import Path

import torch
import torch.nn as nn

from .mesh import Mesh
from .render import Renderer

from .ray_helper import RaycastingImaging
from trimesh import Trimesh

import sys
sys.path.append("../")
from paint3d.utils import inpaint_atlas, save_tensor_image


class TexturedMeshModel(nn.Module):
    def __init__(self, cfg, device=torch.device('cpu')):

        super().__init__()
        self.device = device
        self.cfg = cfg
        self.initial_texture_path = self.cfg.guide.initial_texture
        self.cache_path = Path(self.cfg.log.cache_path) / Path(cfg.guide.shape_path).stem
        self.default_color = self.cfg.render.texture_default_color
        self.force_run_xatlas = self.cfg.guide.force_run_xatlas
        self.max_hits = cfg.render.max_hits

        self.mesh = Mesh(self.cfg.guide.shape_path, self.device, target_scale=self.cfg.guide.shape_scale,
                         mesh_dy=self.cfg.render.look_at_height,
                         remove_mesh_part_names=self.cfg.render.remove_mesh_part_names,
                         remove_unsupported_buffers=self.cfg.render.remove_unsupported_buffers,
                         intermediate_dir=os.path.join(self.cfg.log.exp_path, "convert_results"))
        texture_resolution = self.cfg.guide.texture_resolution
        self.texture_resolution = [texture_resolution[0], texture_resolution[1] * self.mesh.material_num]
        self.renderer = Renderer(render_cfg=self.cfg.render, mesh_face_num=self.mesh.faces.shape[0], device=self.device)
        self.refresh_texture()
        self.vt, self.ft = self.init_texture_map()
        self.face_attributes = kal.ops.mesh.index_vertices_by_faces(self.vt.unsqueeze(0), self.ft.long()).detach()
        # texture map list for texture fusion
        self.texture_list = []
        self.generate_occluded_geometry()

    def init_paint(self):
        if self.initial_texture_path is not None:
            texture_map = Image.open(self.initial_texture_path).convert("RGB").resize(self.texture_resolution)
            texture = torch.Tensor(np.array(texture_map)).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        else:
            texture = torch.ones(1, 3, *self.texture_resolution).to(self.device) * torch.Tensor(
                self.default_color).reshape(1, 3, 1, 1).to(self.device)
        texture_img = nn.Parameter(texture)
        return texture_img

    def refresh_texture(self):
        self.texture_img = self.init_paint()
        self.texture_mask = torch.zeros_like(self.texture_img)
        self.postprocess_edge = torch.zeros_like(self.texture_img)
        self.meta_texture_img = nn.Parameter(torch.zeros_like(self.texture_img))
        self.texture_img_postprocess = None

    def init_texture_map(self):
        cache_path = self.cache_path
        if cache_path is None:
            cache_exists_flag = False
        else:
            vt_cache, ft_cache = cache_path / 'vt.pth', cache_path / 'ft.pth'
            cache_exists_flag = vt_cache.exists() and ft_cache.exists()

        run_xatlas = False
        if self.mesh.vt is not None and self.mesh.ft is not None \
                and self.mesh.vt.shape[0] > 0 and self.mesh.ft.min() > -1:
            vt = self.mesh.vt.to(self.device)
            ft = self.mesh.ft.to(self.device)
        elif cache_exists_flag:
            vt = torch.load(vt_cache).to(self.device)
            ft = torch.load(ft_cache).to(self.device)
        else:
            run_xatlas = True

        if run_xatlas or self.force_run_xatlas:
            import xatlas
            v_np = self.mesh.vertices.cpu().numpy()
            f_np = self.mesh.faces.int().cpu().numpy()
            logger.info(f'running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(self.device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(self.device)
            if cache_path is not None:
                os.makedirs(cache_path, exist_ok=True)
                torch.save(vt.cpu(), vt_cache)
                torch.save(ft.cpu(), ft_cache)
        return vt, ft

    def generate_occluded_geometry(self, faces, texture_init_maps, new_verts_uvs):
        vertices = self.mesh.verts_packed().cpu().numpy()  # (V, 3) shape, move to CPU and convert to numpy
        faces = self.mesh.faces_packed().cpu().numpy()  # (F, 3) shape, move to CPU and convert to numpy

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

    def forward(self, x):
        raise NotImplementedError

    def get_params(self):
        return [self.texture_img, self.meta_texture_img]

    @torch.no_grad()
    def export_mesh(self, path, export_texture_only=False):
        texture_img = self.texture_img.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        texture_img = Image.fromarray((texture_img[0].cpu().detach().numpy() * 255).astype(np.uint8))
        if not os.path.exists(path):
            os.makedirs(path)
        texture_img.save(os.path.join(path, f'albedo.png'))

        if self.texture_img_postprocess is not None:
            texture_img_post = self.texture_img_postprocess.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
            texture_img_post = Image.fromarray((texture_img_post[0].cpu().detach().numpy() * 255).astype(np.uint8))
            os.system("mv {} {}".format(texture_img.save(os.path.join(path, f'albedo.png')),
                                        texture_img.save(os.path.join(path, f'albedo_before.png'))))
            texture_img_post.save(os.path.join(path, f'albedo.png'))

        if export_texture_only: return 0

        v, f = self.mesh.vertices, self.mesh.faces.int()
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]
        vt_np = self.vt.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy()

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'mesh.obj')
        mtl_file = os.path.join(path, f'mesh.mtl')

        logger.info(f'writing obj mesh to {obj_file} with: vertices:{v_np.shape} uv:{vt_np.shape} faces:{f_np.shape}')
        with open(obj_file, "w") as fp:
            fp.write(f'mtllib mesh.mtl \n')

            for v in v_np:     
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            for v in vt_np:    
                fp.write(f'vt {v[0]} {v[1]} \n')

            fp.write(f'usemtl mat0 \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl mat0 \n')
            fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
            fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
            fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
            fp.write(f'Tr 1.000000 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0.000000 \n')
            fp.write(f'map_Kd albedo.png \n')

        if self.mesh.material_cvt is not None:
            logger.info("Postprocess for multiple texture maps or converted mesh~")
            convert_results_dir = os.path.join(path, "convert_results")
            if not os.path.exists(convert_results_dir):
                os.makedirs(convert_results_dir)
            h, w = self.mesh.material_cvt.shape[:2]
            if w % h != 0:
                logger.info("Number of material may be inaccurate, please check manually~")
            for material_id, material in enumerate(np.split(np.array(texture_img), w // h, axis=1)):
                cv2.imwrite(os.path.join(convert_results_dir, "texture_split{}.png".format(material_id)),
                            cv2.cvtColor(material, cv2.COLOR_RGB2BGR))

    def forward_texturing(self, view_target, theta, phi, radius, save_result_dir, view_id=None, verbose=False):
        outputs = self.render(theta=theta, phi=phi, radius=radius)
        uncolored_mask_render = outputs['uncolored_mask']  # bchw, [0,1]
        erode_size = 19
        uncolored_mask_render = torch.from_numpy(
            cv2.erode(uncolored_mask_render[0, 0].detach().cpu().numpy(), np.ones((erode_size, erode_size), np.uint8))).to(
            uncolored_mask_render.device).unsqueeze(0).unsqueeze(0)

        cur_texture_map, cur_texture_mask, weight_map = \
            self.renderer.forward_texturing_render(self.mesh.vertices, self.mesh.faces, self.face_attributes,
                                                   theta=theta, phi=phi, radius=radius, view_target=view_target,
                                                   uncolored_mask=uncolored_mask_render,
                                                   texture_dims=self.texture_resolution)
        if verbose:
            save_tensor_image(view_target, os.path.join(save_result_dir, f"_view_{view_id}_view_target.png"))
            save_tensor_image(uncolored_mask_render.repeat(1,3,1,1),
                              os.path.join(save_result_dir, f"_view_{view_id}_uncolored_mask_render.png"))
            save_t = view_target * uncolored_mask_render.repeat(1,3,1,1)
            save_tensor_image(save_t, os.path.join(save_result_dir, f"_view_{view_id}_uncolored_masked_img.png"))
            save_tensor_image(cur_texture_map, os.path.join(save_result_dir, f"_view_{view_id}_cur_texture_map.png"))
            save_tensor_image(cur_texture_mask, os.path.join(save_result_dir, f"_view_{view_id}_cur_texture_mask.png"))
            save_tensor_image(weight_map, os.path.join(save_result_dir, f"_view_{view_id}_weight_map.png"))

        updated_texture_map = cur_texture_map * cur_texture_mask + self.texture_img * (1 - cur_texture_mask)
        save_tensor_image(updated_texture_map, os.path.join(save_result_dir, f"_view_{view_id}_texture_map.png"))
        self.texture_img = nn.Parameter(updated_texture_map)


    def texture_fusion(self):
        texture, weight_map = self.texture_list[0]
        texture_fusion = torch.zeros_like(texture, dtype=torch.float32)

        weight_maps = torch.cat([weight for _, weight in self.texture_list], dim=1)
        fused_weights = torch.unbind(torch.softmax(weight_maps * 10, dim=1), dim=1)
        for i in range(len(self.texture_list)):
            texture = self.texture_list[i][0]
            weight_map = fused_weights[i].unsqueeze(1)
            texture_fusion += texture * weight_map

        default_texture = torch.ones(1, 3, *self.texture_resolution).to(self.device) * torch.Tensor(
            self.default_color).reshape(1, 3, 1, 1).to(self.device)
        mask = torch.zeros_like(texture_fusion, dtype=torch.float32)
        mask[texture_fusion > 0 ] = 1
        texture_fusion = texture_fusion * mask + (1 - mask) * default_texture
        return texture_fusion

    def empty_texture_cache(self):
        self.texture_list = []

    def texture_postprocess(self):
        texture_img_npy = self.texture_img.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        texture_img_npy = (texture_img_npy[0].cpu().detach().numpy() * 255).astype(np.uint8)

        append_mask_edge = self.postprocess_edge.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        append_mask_edge = (append_mask_edge[0].cpu().detach().numpy() * 255).astype(np.uint8)
        texture_img_npy_inpaint = inpaint_atlas(texture_img_npy, append_mask_edge)
        self.texture_img_postprocess = nn.Parameter(torch.from_numpy(texture_img_npy_inpaint/255.0).unsqueeze(0).permute(0, 3, 1, 2))

    def render(self, theta=None, phi=None, radius=None, use_meta_texture=False, render_cache=None, dims=None):
        if render_cache is None:
            assert theta is not None and phi is not None and radius is not None
        if use_meta_texture:
            texture_img = self.meta_texture_img
        else:
            texture_img = self.texture_img

        rgb, depth, mask, uncolored_mask, normals, render_cache = \
            self.renderer.render_single_view_texture(self.mesh.vertices, self.mesh.faces, self.face_attributes,
                                                     texture_img, theta=theta, phi=phi, radius=radius,
                                                     render_cache=render_cache, dims=dims,
                                                     texture_default_color=self.default_color)
        if not use_meta_texture:
            rgb = rgb.clamp(0, 1)

        return {'image': rgb, 'mask': mask.detach(), 'uncolored_mask': uncolored_mask, 'depth': depth,
                'normals': normals, 'render_cache': render_cache, 'texture_map': texture_img}


    def UV_pos_render(self):
        UV_pos = self.renderer.UV_pos_render(self.mesh.vertices, self.mesh.faces, self.face_attributes,
                                    texture_dims=self.texture_resolution)
        return UV_pos
