import tempfile
import os
from typing import Union

import folder_paths
import trimesh
import pymeshlab



def bpt_remesh(self, mesh: trimesh.Trimesh, verbose: bool = False, with_normal: bool = True, temperature: float = 0.5, batch_size: int = 1, pc_num: int = 4096, seed: int = 1234, samples: int = 50000):
        from .model import data_utils
        from .model.model import MeshTransformer
        from .model.serializaiton import BPT_deserialize
        from .utils import sample_pc, joint_filter

        pc_normal = sample_pc(mesh, pc_num=pc_num, with_normal=with_normal, seed=seed, samples=samples)

        pc_normal = pc_normal[None, :, :] if len(pc_normal.shape) == 2 else pc_normal

        from torch.serialization import add_safe_globals
        from deepspeed.runtime.fp16.loss_scaler import LossScaler
        from deepspeed.runtime.zero.config import ZeroStageEnum
        from deepspeed.utils.tensor_fragment import fragment_address

        add_safe_globals([LossScaler, fragment_address, ZeroStageEnum])

        model = MeshTransformer(mode='faces')

        comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        model_path = os.path.join(comfy_path, "models", "bpt", "bpt-8-16-500m.pt")
        print(f"BPT Model Path: {model_path} \n")
        model.load(model_path)
        model = model.eval()
        model = model.half()
        model = model.cuda()

        import torch
        pc_tensor = torch.from_numpy(pc_normal).cuda().half()
        if len(pc_tensor.shape) == 2:
            pc_tensor = pc_tensor.unsqueeze(0)

        codes = model.generate(
            pc=pc_tensor,
            filter_logits_fn=joint_filter,
            filter_kwargs=dict(k=50, p=0.95),
            return_codes=True,
            temperature=temperature,
            batch_size=batch_size,
        )
        
        coords = []
        try:
            for i in range(len(codes)):
                code = codes[i]
                code = code[code != model.pad_id].cpu().numpy()
                vertices = BPT_deserialize(
                    code,
                    block_size=model.block_size,
                    offset_size=model.offset_size,
                    use_special_block=model.use_special_block,
                )
                coords.append(vertices)
        except:
            coords.append(np.zeros(3, 3))

        vertices = coords[i]
        faces = torch.arange(1, len(vertices) + 1).view(-1, 3)

        # Move to CPU
        faces = faces.cpu().numpy()

        del model

        return data_utils.to_mesh(vertices, faces, transpose=False, post_process=True)

        
class BptMesh:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, str],
        temperature: float = 0.5,
        batch_size: int = 1,
        with_normal: bool = True,
        verbose: bool = False,
        pc_num: int = 4096,
        seed: int = 1234,
        samples: int = 50000
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        mesh = bpt_remesh(self, mesh=mesh, temperature=temperature, batch_size=batch_size, with_normal=with_normal, pc_num=pc_num, seed=seed, samples=samples)
        return mesh

