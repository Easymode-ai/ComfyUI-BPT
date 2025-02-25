import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import numpy as np
import json
import trimesh as Trimesh
from tqdm import tqdm

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale


script_directory = os.path.dirname(os.path.abspath(__file__))

class ComfyProgressCallback:
    def __init__(self, total_steps):
        self.pbar = ProgressBar(total_steps)
        
    def __call__(self, pipe, i, t, callback_kwargs):
        self.pbar.update(1)
        return {
            "latents": callback_kwargs["latents"],
            "prompt_embeds": callback_kwargs["prompt_embeds"],
            "negative_prompt_embeds": callback_kwargs["negative_prompt_embeds"]
        }

class TrimeshLoad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "load_path": ("STRING", {"default": "", "tooltip": "The path of the mesh to load."}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    OUTPUT_TOOLTIPS = ("The model of the mesh to load.",)
    
    FUNCTION = "load"
    CATEGORY = "BPT"
    DESCRIPTION = "Loads a model from the given path."

    def load(self, load_path, file_format):
        
        trimesh = Trimesh.load(load_path, force="mesh", format=file_format)
        
        return (trimesh,)
    
class TrimeshBPT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "enable_bpt": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffff}),
                "temperature": ("FLOAT", {"default": 0.5}),
                "pc_num": ("INT", {"default": 4096, "min": 1024, "max": 8192, "step": 1024}),
                "samples": ("INT", {"default": 100000}),
                "enable_reduce_faces": ("BOOLEAN", {"default": True}),
                "target_num_faces": ("INT", {"default": 50000}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "bpt"
    CATEGORY = "BPT"
    DESCRIPTION = "BPT using bpt: https://github.com/whaohan/bpt (Trimesh in/out)"
    
    def bpt(self, trimesh, enable_bpt, temperature, pc_num, seed, samples, enable_reduce_faces, target_num_faces):
        mm.unload_all_models()
        mm.soft_empty_cache()
        new_mesh = trimesh.copy()
        if enable_bpt:
            from .bpt import BptMesh

            if enable_bpt:
                if enable_reduce_faces:
                      if len(new_mesh.faces) > target_num_faces:
                            try:
                                import pyfqmr
                            except ImportError:
                                raise ImportError("pyfqmr not found. Please install it using 'pip install pyfqmr' https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction")
                            import pymeshlab
                            from typing import Union
                            import tempfile

                            def trimesh2pymeshlab(mesh: Trimesh.Trimesh) -> pymeshlab.MeshSet:
                                with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
                                    temp_file_name = temp_file.name

                                if isinstance(mesh, Trimesh.scene.Scene):
                                    combined_mesh = None
                                    for idx, obj in enumerate(mesh.geometry.values()):
                                        combined_mesh = obj if idx == 0 else combined_mesh + obj
                                    mesh = combined_mesh

                                print("Exporting to temporary file...\n")
                                mesh.export(temp_file_name)  # ✅ Ensure mesh is actually written

                                print("Loading into PyMeshLab...\n")
                                ms = pymeshlab.MeshSet()
                                ms.load_new_mesh(temp_file_name)  # ✅ Load into PyMeshLab

                                os.remove(temp_file_name)  # ✅ Clean up

                                return ms

                            def import_mesh(mesh: Union[pymeshlab.MeshSet, Trimesh.Trimesh, str]) -> pymeshlab.MeshSet:
                                if isinstance(mesh, str):
                                    print("Loading mesh from file...\n")
                                    ms = pymeshlab.MeshSet()
                                    ms.load_new_mesh(mesh)
                                    return ms

                                if isinstance(mesh, (Trimesh.Trimesh, Trimesh.scene.Scene)):
                                    print("Converting Trimesh to PyMeshLab...\n")
                                    return trimesh2pymeshlab(mesh)  # ✅ Use correct conversion function

                                if isinstance(mesh, pymeshlab.MeshSet):
                                    return mesh  # Already correct type

                                raise ValueError("Unsupported mesh type")
                            
                            def reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int = 200000):
                                print("reducer...\n")
                                mesh.apply_filter(
                                    "meshing_decimation_quadric_edge_collapse",
                                    targetfacenum=max_facenum,
                                    qualitythr=5.0,
                                    preserveboundary=False,
                                    boundaryweight=3,
                                    preservenormal=True,
                                    preservetopology=True,
                                    autoclean=False
                                )
                                return mesh
                            
                            ms = import_mesh(new_mesh)
                            ms = reduce_face(ms, max_facenum=target_num_faces)
                            current_mesh = ms.current_mesh()

                            new_mesh = Trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())
                            

                            mesh_simplifier = pyfqmr.Simplify()
                            mesh_simplifier.setMesh(new_mesh.vertices, new_mesh.faces)
                            mesh_simplifier.simplify_mesh(
                                target_count=target_num_faces, 
                                aggressiveness=7,
                                update_rate=5,
                                max_iterations=100,
                                preserve_border=True, 
                                verbose=True,
                                lossless=False,
                                threshold_lossless=1e-3
                                )
                            new_mesh.vertices, new_mesh.faces, _ = mesh_simplifier.getMesh()

                new_mesh = BptMesh()(new_mesh, with_normal=True, temperature=temperature, batch_size=1, pc_num=pc_num, verbose=False, seed=seed, samples=samples)
                mm.unload_all_models()
                mm.soft_empty_cache()

        return (new_mesh, )
    
class TrimeshSave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/BPT_"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "BPT"
    OUTPUT_NODE = True

    def save(self, trimesh, filename_prefix, file_format):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        full_output_folder = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        full_output_folder.parent.mkdir(exist_ok=True)
        print(f"Saved to disk: {full_output_folder}")
        trimesh.export(full_output_folder, file_type=file_format)        
        return ()

class TrimeshPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("relative_path",)
    FUNCTION = "savepreview"
    CATEGORY = "BPT"
    OUTPUT_NODE = True

    def savepreview(self, trimesh, file_format):
        # Ensure comfy_path is a Path object
        filename_prefix = "PBTtemp_"
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)

        temp_file = Path(full_output_folder, f'hy3dtemp_{counter:05}_.{file_format}')
        trimesh.export(temp_file, file_type=file_format)
        relative_path = Path(subfolder) / f'hy3dtemp_{counter:05}_.{file_format}'
        
        return (str(relative_path), )
    
NODE_CLASS_MAPPINGS = {
    "TrimeshBPT": TrimeshBPT,
    "TrimeshLoad": TrimeshLoad,
    "TrimeshSave": TrimeshSave,
    "TrimeshPreview": TrimeshPreview,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrimeshBPT": "BPT",
    "TrimeshLoad": "Trimesh Load",
    "TrimeshSave": "Trimesh Save",
    "TrimeshPreview": "Trimesh Preview"
    }