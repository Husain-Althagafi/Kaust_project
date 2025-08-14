import torch
from .vision_transformer import vit_base, VisionTransformer
import open_clip
import os
    
def load_open_clip_as_dino_generic(model_name: str, pretrained: str, cache_dir: str):
    """
    Loads any OpenCLIP model and adapts its weights to a DINO-style VisionTransformer.
    This is a more robust, generic version.
    """
    print(f"Loading OpenCLIP model: {model_name} with pretrained weights: {pretrained}")
    
    # 1. Load the original OpenCLIP model to get its config and weights
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, cache_dir=cache_dir
    )


    # 2. Get the original state dict and extract the visual part
    original_sd = clip_model.state_dict()
    visual_sd = {}
    for k, v in original_sd.items():
        if k.startswith('visual.'):
            visual_sd[k.replace('visual.', '')] = v

    # 3. Create the kwargs for our DINO-style VisionTransformer.
    #    We extract these directly from the loaded OpenCLIP model's architecture.
    
    visual_model = clip_model.visual
    num_heads = clip_model.visual.transformer.resblocks[0].attn.num_heads

    kwargs = {
        'patch_size': visual_model.conv1.kernel_size[0],
        'embed_dim': visual_model.ln_pre.normalized_shape[0], # or visual_model.conv1.out_channels
        'depth': len(visual_model.transformer.resblocks),
        'num_heads': num_heads,
        'mlp_ratio': 4.0, # Standard for ViTs, but can be inferred from mlp if needed
    }
    
    print("Inferred DINO ViT kwargs:", kwargs) # Good for debugging
    
    # 4. Create an instance of our DINO-style ViT with the correct architecture
    dino_vit = VisionTransformer(**kwargs)
    
    # 5. Load the adapted weights
    # We use strict=False because the DINO ViT won't have the final 'proj' layer
    # that CLIP uses for its contrastive loss.
    missing_keys, unexpected_keys = dino_vit.load_state_dict(visual_sd, strict=False)
    print("Keys missing in DINO ViT:", missing_keys)
    print("Keys unexpected in DINO ViT:", unexpected_keys)

    # 6. Handle the projection layer
    # The 'proj' is the final output layer of the CLIP model. We need to handle it separately.
    proj = original_sd.get('visual.proj', None)
    if proj is None:
        print("Warning: Could not find 'visual.proj' in state_dict.")

    # 7. Final adjustments to match DINO's behavior
    dino_vit.pos_drop = torch.nn.LayerNorm(dino_vit.embed_dim)
    for m in dino_vit.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.eps = 1e-5
            
    return dino_vit, proj
    

def load_open_clip_as_dino(patch_size, load_dir="/ibex/user/thagafhh/data/dreamsim_data/models", l14=False, h14=False):
    if h14:
        # --- NEW LOGIC FOR ViT-H-14 ---
        # We call our new generic function
        return load_open_clip_as_dino_generic(
            model_name='ViT-H-14',
            pretrained='laion2b_s32b_b79k',
            cache_dir=load_dir
        )
    if l14:
        sd = torch.load(os.path.join(load_dir, 'open_clipl14_as_dino_vitl/checkpoint.pt'), map_location='cpu', weights_only=False)
        dino_vit = VisionTransformer(**sd['kwargs'])
        sd = sd['state_dict']
    else:
        dino_vit = vit_base(patch_size=patch_size)
        sd = torch.load(os.path.join(load_dir, f'open_clip_vitb{patch_size}_pretrain.pth.tar'), weights_only=True)['state_dict']

    dino_vit.pos_drop = torch.nn.LayerNorm(dino_vit.embed_dim)
    proj = sd.pop('proj')
    dino_vit.load_state_dict(sd)
    for m in dino_vit.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.eps = 1e-5

    return dino_vit, proj

    
@torch.no_grad()
def _sanity_check(patch_size, l14=False, h14=False):
    from PIL import Image

    if h14:
        dino_vit, proj = load_open_clip_as_dino(patch_size, h14=True)
        clip_all, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir='.'
            )
    if l14:
        dino_vit, proj = load_open_clip_as_dino(patch_size, l14=True)
        clip_all, _, preprocess = open_clip.create_model_and_transforms(f'ViT-L-{patch_size}',
                                                                        pretrained='laion400m_e31', cache_dir=".")
    else:
        dino_vit, proj = load_open_clip_as_dino(patch_size)
        clip_all, _, preprocess = open_clip.create_model_and_transforms(f'ViT-B-{patch_size}',
                                                                        pretrained='laion400m_e31', cache_dir=".")

    x = preprocess(Image.open('images/img_a_1.png'))[None, ...]

    # intermidiate representations
    cpre = []

    def clip_hook(module, x, y):
        if len(cpre) == 0:
            cpre.append(y)
        else:
            cpre.append(y.transpose(0, 1))  # batch and seq_len dimensions are flipped in CLIP

    clip_all.visual.ln_pre.register_forward_hook(clip_hook)  # checks
    [rb.register_forward_hook(clip_hook) for rb in clip_all.visual.transformer.resblocks]

    vpre = []

    def vit_hook(module, x, y):
        vpre.append(y)

    dino_vit.pos_drop.register_forward_hook(vit_hook)  # checks
    [b.register_forward_hook(vit_hook) for b in dino_vit.blocks]

    # forward
    ce = clip_all.encode_image(x)

    ve = dino_vit(x) @ proj

    for i, (c, v) in enumerate(zip(cpre, vpre)):
        delta = (c - v).abs()
        assert torch.isclose(c, v).all(), f'layer {i}:\ndelta={delta}\n{torch.isclose(c, v)}'
        print(f'{i}: {delta.abs().max()}')

    delta = (ce - ve).abs()
    assert torch.isclose(ce, ve).all(), f'final rep:\ndelta={delta}\n{torch.isclose(ce, ve)}'
    print(f'emb space: {delta.abs().max()}')


if __name__ == '__main__':
    # _sanity_check(patch_size=32)
    # _sanity_check(patch_size=14, l14=True)
    _sanity_check(patch_size=14, h14=True)

    #_sanity_check(patch_size=16)
    #_sanity_check(patch_size=14)
