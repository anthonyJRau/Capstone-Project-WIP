import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, MllamaForConditionalGeneration
from tqdm import tqdm
import pandas as pd
import ast
import sys
import os
from transformers import BitsAndBytesConfig

def create_annotated_image(img_path, bbox, label='Object'):
    """Create image with bounding box annotation"""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Parse bbox
    if isinstance(bbox, str):
        x_min, y_min, x_max, y_max = ast.literal_eval(bbox)
    else:
        x_min, y_min, x_max, y_max = bbox
    
    # Draw rectangle and label
    draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((x_min, y_min - 20), label, fill='red', font=font)
    
    return img

def save_results(val_data, filename='llama_results_hand_labeled_90B.feather'):
    """Save with error handling"""
    try:
        val_data.to_feather(filename)
        return True
    except Exception as e:
        print(f"\nError saving to {filename}: {e}")
        # Try backup save
        try:
            val_data.to_csv(filename.replace('.feather', '_backup.csv'))
            print(f"Saved backup CSV instead")
            return True
        except Exception as e2:
            print(f"Failed to save backup: {e2}")
            return False

def main():
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Load data
    print("\nLoading data...")
    val_data = pd.read_feather('hand_labeled_data.feather')
    val_data = val_data.dropna()
    val_data = val_data[val_data['new_label']!=2].reset_index(drop=True)
    
    # Check if we're resuming
    try:
        existing = pd.read_feather('NOPE.feather')
        if 'llama-response' in existing.columns:
            val_data = existing
            # Count actual successful responses (not errors)
            successful_mask = val_data['llama-response'].apply(
                lambda x: pd.notna(x) and x not in ["ERROR", "SHAPE_ERROR", None]
            )
            start_idx = successful_mask.sum()
            print(f"Found {start_idx} successful responses, resuming from index {start_idx}")
        else:
            val_data['llama-response'] = None
            start_idx = 0
    except:
        val_data['llama-response'] = None
        start_idx = 0
    
    dataroot = 'data/nuscenes/datasets/v1.0-trainval/'
    
    # Load model with 4-bit quantization (more stable for multi-GPU)
    print("\nLoading model with 4-bit quantization on multi-GPU...")
    model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",  # Let it distribute across GPUs
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    print("\nModel loaded successfully!")
    print("GPU memory usage:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    # Process images
    successful = 0
    failed = 0
    consecutive_failures = 0
    
    # Save initial state
    print("\nSaving initial state...")
    save_results(val_data)
    
    print(f"\nProcessing {len(val_data) - start_idx} images starting from index {start_idx}...\n")
    
    for i in tqdm(range(start_idx, len(val_data)), desc="Processing"):
        try:
            # Load and annotate image
            img_path = f"{dataroot}/{val_data.iloc[i]['filename']}"
            bbox = val_data.iloc[i]['bbox_2d']
            class_name = 'bicycle'
            
            img = create_annotated_image(img_path, bbox, label=class_name)
            
            # Ensure reasonable image size
            if max(img.size) > 1120:
                ratio = 1120 / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Create prompt
            prompt = f"""Is the {class_name} with the red box stationary or moving? 
            If the {class_name} is in the road and there is a rider on it then it is moving.
            If the {class_name} is not in the road and there is not a rider on it, it is laying down or other then it is stationary.
            With this context, answer ONLY 'stationary' or 'moving'.
            """
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=img, text=text, return_tensors="pt")
            
            # Move to first GPU
            inputs = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate with inference mode
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
            
            # Decode only the new tokens
            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = output[0][prompt_length:]
            response = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            val_data.at[i, 'llama-response'] = response
            successful += 1
            consecutive_failures = 0  # Reset counter
            
            # Print first few successes to verify
            if successful <= 5:
                tqdm.write(f"✓ Sample response {successful}: '{response}'")
            
            # Cleanup
            del inputs, output, generated_ids
            torch.cuda.empty_cache()
            
            # Save every 10 iterations
            if (i + 1) % 10 == 0:
                if save_results(val_data):
                    tqdm.write(f"✓ Saved at {i+1}/{len(val_data)} (Success: {successful}, Failed: {failed})")
            
        except Exception as e:
            error_msg = str(e)
            tqdm.write(f"\n❌ Error at index {i}: {error_msg[:200]}")
            val_data.at[i, 'llama-response'] = f"ERROR: {error_msg[:50]}"
            failed += 1
            consecutive_failures += 1
            
            # Save immediately on error
            save_results(val_data)
            torch.cuda.empty_cache()
            
            # If we get 10 consecutive errors, something is fundamentally wrong
            if consecutive_failures >= 10:
                print(f"\n❌ {consecutive_failures} consecutive errors. Something is fundamentally broken.")
                print("Saving and exiting...")
                save_results(val_data)
                sys.exit(1)
    
    # Final save
    print("\nFinal save...")
    save_results(val_data)
    print(f"\n✅ Done! Success: {successful}, Failed: {failed}")
    
    # Print final GPU memory usage
    print("\nFinal GPU memory usage:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {allocated:.2f} GB")

if __name__ == "__main__":
    main()