import torch
import os
import sys

# Add dreamsim to path so we can import its training class
dreamsim_path = '/home/thagafhh/work/dreamsim/dreamsim'
if dreamsim_path not in sys.path:
    sys.path.append(dreamsim_path)

from training.train import LightningPerceptualModel

def export_native_weights(checkpoint_path: str, output_path: str):
    """
    Loads a .ckpt file, performs the full weight surgery (renaming AND reshaping),
    and saves the final, native-compatible state_dict.
    """
    print("Loading Lightning checkpoint...")
    lightning_model = LightningPerceptualModel.load_from_checkpoint(checkpoint_path, map_location='cpu')

    print("Merging LoRA weights...")
    merged_perceptual_model = lightning_model.perceptual_model.merge_and_unload()
    final_dino_vit = merged_perceptual_model.extractor_list[0].model
    
    print("Extracting DINO-like state dictionary...")
    dino_state_dict = final_dino_vit.state_dict()
    
    native_state_dict = {}
    print("Starting weight conversion (renaming and reshaping)...")

    for dino_key, dino_tensor in dino_state_dict.items():
        native_key = dino_key
        native_tensor = dino_tensor

        # --- Part 1: Reshaping ---
        # The DINO implementation has an extra dimension for these parameters.
        # We must remove it.
        if dino_key in ["cls_token", "pos_embed"]:
            if native_tensor.dim() > 2: # e.g., shape is [1, 257, 1280]
                print(f"  Reshaping '{dino_key}': {native_tensor.shape} -> {native_tensor.squeeze(0).shape}")
                native_tensor = native_tensor.squeeze(0)
            if native_tensor.dim() > 1 and dino_key == "cls_token": # e.g., shape is [1, 1280]
                 print(f"  Reshaping '{dino_key}': {native_tensor.shape} -> {native_tensor.squeeze(0).shape}")
                 native_tensor = native_tensor.squeeze(0)


        # --- Part 2: Renaming ---
        # A simple chain of replacements is robust.
        native_key = native_key.replace('blocks', 'transformer.resblocks')
        native_key = native_key.replace('patch_embed.proj', 'conv1')
        native_key = native_key.replace('pos_embed', 'positional_embedding')
        native_key = native_key.replace('cls_token', 'class_embedding')
        native_key = native_key.replace('norm.', 'ln_post.') # Note the dot to be specific
        native_key = native_key.replace('norm1', 'ln_1')
        native_key = native_key.replace('norm2', 'ln_2')
        native_key = native_key.replace('attn.qkv', 'attn.in_proj')
        native_key = native_key.replace('attn.proj', 'attn.out_proj')
        native_key = native_key.replace('mlp.fc1', 'mlp.c_fc')
        native_key = native_key.replace('mlp.fc2', 'mlp.c_proj')
        
        if dino_key != native_key:
            print(f"  Renaming '{dino_key}' -> '{native_key}'")

        native_state_dict[native_key] = native_tensor

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving final native-compatible weights to: {output_path}")
    torch.save(native_state_dict, output_path)
    print("--- Export complete! ---")


if __name__ == '__main__':
    CHECKPOINT_FILE_PATH = '/ibex/user/thagafhh/data/dreamsim_data/logs/finetune__dino-like_clip_h14_mlp_head_open_clip_vith14_cls_lora_lr_0.001_batchsize_16_wd_0.0_hiddensize_512_margin_0.01_lorar_8_loraalpha_0.1_loradropout_0.1/dreamsim_training/amv064co/checkpoints/epoch=09.ckpt'
    # Save to a new file to be sure we're using the latest version
    EXPORTED_WEIGHTS_PATH = "/ibex/user/thagafhh/data/dreamsim_data/converted_weights/exported_native_weights_v3_final.pth"

    export_native_weights(CHECKPOINT_FILE_PATH, EXPORTED_WEIGHTS_PATH)