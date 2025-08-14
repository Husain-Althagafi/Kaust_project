import torch
import open_clip

def load_human_aligned_teacher(
    exported_weights_path: str,
    cache_dir: str = "/ibex/user/thagafhh/data/dreamsim_data/models"
) -> torch.nn.Module:
    """
    Loads a native OpenCLIP ViT-H-14 model and applies the final, pre-converted,
    human-aligned weights to it.

    This function is fully standalone and has no dependencies on DreamSim.

    Args:
        exported_weights_path (str): The full path to your clean .pth weight file.
        cache_dir (str): Directory where the base OpenCLIP models are downloaded.

    Returns:
        A ready-to-use, human-aligned image encoder model.
    """
    print("--- Loading Final Human-Aligned Teacher Model ---")

    # 1. Load the NATIVE OpenCLIP base model. This provides the correct
    #    architecture that your weights were converted to fit.
    print(f"Loading native 'ViT-H-14' base architecture from {cache_dir}...")
    native_clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained=False,  # Set to False! We don't need the original weights, just the architecture.
        cache_dir=cache_dir
    )
    native_model_body = native_clip_model.visual
    proj_head = native_clip_model.visual.proj

    # 2. Load your clean, exported state dictionary.
    print(f"Loading pre-converted weights from: {exported_weights_path}")
    state_dict = torch.load(exported_weights_path, map_location='cpu')

    # 3. Load the weights into the model's body.
    #    This should work perfectly now because the keys have been renamed.
    native_model_body.load_state_dict(state_dict, strict=False)
    
    # 4. Wrap the final components in a clean, easy-to-use encoder class.
    class FinalEncoder(torch.nn.Module):
        def __init__(self, model_body):
            super().__init__()
            # We only need to store the main model body now
            self.model_body = model_body

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # --- THE FIX ---
            # The .forward() method of the OpenCLIP VisionTransformer already
            # applies the internal projection head. We just return its output.
            embedding_1024 = self.model_body(x)
            # --- END OF FIX ---
            return embedding_1024
            
    # We only need to pass the model body to the encoder now
    final_model = FinalEncoder(native_model_body)
    
    print("--- Human-aligned teacher model ready for use! ---")
    return final_model