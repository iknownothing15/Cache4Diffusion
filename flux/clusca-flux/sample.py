import os
import pickle
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from einops import rearrange
from PIL import ExifTags, Image
from transformers import pipeline
from tqdm import tqdm

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, denoise_test_FLOPs
from flux.ideas import denoise_cache
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85  # NSFW score threshold


@dataclass
class SamplingOptions:
    prompts: list[str]          # List of prompts
    width: int                  # Image width
    height: int                 # Image height
    num_steps: int              # Number of sampling steps
    guidance: float             # Guidance value
    seed: int | None            # Random seed
    num_images_per_prompt: int  # Number of images generated per prompt
    batch_size: int             # Batch size (batching of prompts)
    model_name: str             # Model name
    output_dir: str             # Output directory
    add_sampling_metadata: bool # Whether to add metadata
    use_nsfw_filter: bool       # Whether to enable NSFW filter
    test_FLOPs: bool            # Whether in FLOPs test mode (no actual image generation)
    mode: str = 'Taylor'        # Pipeline mode: embed / infer / cache strategy


def embed_prompts(opts: SamplingOptions, embed_path: str):
    """Step 1: Encode all prompts with T5/CLIP and save to a pickle file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Embed Mode] Encoding prompts and saving to: {embed_path}")

    # Load only T5 and CLIP (no model, no AE)
    t5 = load_t5(device, max_length=256 if opts.model_name == "flux-schnell" else 512)
    clip = load_clip(device)

    # Set random seed
    base_seed = opts.seed if opts.seed is not None else torch.randint(0, 2**32, (1,)).item()

    prompts = opts.prompts
    num_prompt_batches = (len(prompts) + opts.batch_size - 1) // opts.batch_size
    all_inputs = []
    idx = 0

    for batch_idx in tqdm(range(num_prompt_batches), desc="Encoding prompts"):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(prompts))
        batch_prompts = prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        for image_idx in range(opts.num_images_per_prompt):
            seed = base_seed + idx
            idx += num_prompts_in_batch

            x = get_noise(
                num_prompts_in_batch, opts.height, opts.width,
                device=device, dtype=torch.bfloat16, seed=seed,
            )

            inp = prepare(t5, clip, x, prompt=batch_prompts)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1],
                                     shift=(opts.model_name != "flux-schnell"))

            # Move tensors to CPU for pickling
            inp_cpu = {k: v.cpu() for k, v in inp.items()}
            all_inputs.append({
                'inp': inp_cpu,
                'timesteps': timesteps,
                'seed': seed,
                'prompts': list(batch_prompts),
                'num_prompts_in_batch': num_prompts_in_batch,
            })

    # Save to pickle
    os.makedirs(os.path.dirname(embed_path) if os.path.dirname(embed_path) else '.', exist_ok=True)
    with open(embed_path, 'wb') as f:
        pickle.dump({
            'inputs': all_inputs,
            'width': opts.width,
            'height': opts.height,
            'num_steps': opts.num_steps,
            'guidance': opts.guidance,
            'model_name': opts.model_name,
        }, f)
    print(f"[Embed Mode] Done. {len(all_inputs)} entries saved to {embed_path}")


def infer_from_embeddings(opts: SamplingOptions, model_kwargs: dict, embed_path: str):
    """Step 2: Load precomputed embeddings and run denoising + decoding."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Infer Mode] Loading embeddings from: {embed_path}")

    with open(embed_path, 'rb') as f:
        embed_data = pickle.load(f)

    all_inputs = embed_data['inputs']
    # Adopt saved config for unspecified / default values
    if opts.width == 1360:
        opts.width = embed_data.get('width', opts.width)
    if opts.height == 768:
        opts.height = embed_data.get('height', opts.height)
    if opts.num_steps is None:
        opts.num_steps = embed_data.get('num_steps', opts.num_steps)
    if opts.guidance == 3.5:
        opts.guidance = embed_data.get('guidance', opts.guidance)

    # Optional NSFW classifier
    nsfw_classifier = None
    if opts.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification",
            model="/root/autodl-tmp/pretrained_models/Falconsai/nsfw_image_detection",
            device=device,
        )

    # Ensure width/height are multiples of 16
    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    cache_mode = model_kwargs.get('cache_mode', 'Taylor')
    opts.output_dir = os.path.join(
        opts.output_dir,
        f"{model_kwargs['fresh_threshold']}-{model_kwargs['max_order']}",
        f"{cache_mode}",
        f"{model_kwargs['cluster_num']}",
        f"{model_kwargs['cluster_method']}",
        f"{model_kwargs['k']}",
        f"{model_kwargs['propagation_ratio']}",
    )
    model_kwargs['topk'] = model_kwargs.get('k', 1)
    print("generating images in:", opts.output_dir)
    output_name = os.path.join(opts.output_dir, "img_{idx}.jpg")
    os.makedirs(opts.output_dir, exist_ok=True)

    # Load model and AE only (no T5/CLIP needed)
    model = load_flow_model(opts.model_name, device=device)
    ae = load_ae(opts.model_name, device=device)

    total_images = sum(entry['num_prompts_in_batch'] for entry in all_inputs)
    progress_bar = tqdm(total=total_images, desc="Generating images")
    global_idx = 0

    for data in all_inputs:
        inp = {k: v.to(device) for k, v in data['inp'].items()}
        timesteps = data['timesteps']
        batch_prompts = data['prompts']
        batch_size = data['num_prompts_in_batch']

        with torch.no_grad():
            infer_kwargs = dict(model_kwargs)
            infer_kwargs['mode'] = cache_mode
            x = denoise_cache(model, **inp, timesteps=timesteps,
                              guidance=opts.guidance, model_kwargs=infer_kwargs)

            x = unpack(x.float(), opts.height, opts.width)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                x = ae.decode(x)

        # Convert to PIL and save
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x, "b c h w -> b h w c")

        for i in range(batch_size):
            img_array = x[i]
            img = Image.fromarray((127.5 * (img_array + 1.0)).cpu().byte().numpy())

            nsfw_score = 0.0
            if opts.use_nsfw_filter:
                nsfw_result = nsfw_classifier(img)
                nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0)

            if nsfw_score < NSFW_THRESHOLD:
                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                exif_data[ExifTags.Base.Model] = opts.model_name
                if opts.add_sampling_metadata:
                    exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]
                fn = output_name.format(idx=global_idx + i)
                img.save(fn, exif=exif_data, quality=95, subsampling=0)
            else:
                print("Generated image may contain inappropriate content, skipped.")

            progress_bar.update(1)

        global_idx += batch_size

    progress_bar.close()
    print(f"[Infer Mode] Done. Generated {global_idx} images.")


def main(opts: SamplingOptions, model_kwargs: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve embed_path default
    embed_path = model_kwargs.get('embed_path', None)
    if embed_path is None:
        embed_path = os.path.join(opts.output_dir, 'embeddings.pkl')

    # --- Dispatch: embed / infer / full pipeline ---
    if opts.mode == 'embed':
        embed_prompts(opts, embed_path)
        return

    if opts.mode == 'infer':
        infer_from_embeddings(opts, model_kwargs, embed_path)
        return

    # ==================== ORIGINAL FULL PIPELINE (other modes) ====================

    # Optional NSFW classifier
    if opts.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification",
            model="/root/autodl-tmp/pretrained_models/Falconsai/nsfw_image_detection",
            device=device
        )
    else:
        nsfw_classifier = None

    # Load model
    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")

    if opts.num_steps is None:
        opts.num_steps = 4 if model_name == "flux-schnell" else 50

    # Ensure width and height are multiples of 16
    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    opts.output_dir = os.path.join(opts.output_dir, 
                                    f"{model_kwargs['fresh_threshold']}-{model_kwargs['max_order']}",
                                    f"{model_kwargs['mode']}",
                                    f"{model_kwargs['cluster_num']}",
                                    f"{model_kwargs['cluster_method']}",
                                    f"{model_kwargs['k']}",
                                    f"{model_kwargs['propagation_ratio']}"
                                    )
    print("generating images in:", opts.output_dir)
    # Set output directory and index
    output_name = os.path.join(opts.output_dir, f"img_{{idx}}.jpg")
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    idx = 0  # Image index

    # Initialize model components
    torch_device = device

    # Load T5 and CLIP models to GPU
    t5 = load_t5(torch_device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(torch_device)

    # Load model to GPU
    model = load_flow_model(model_name, device=torch_device)
    ae = load_ae(model_name, device=torch_device)

    # Set random seed
    if opts.seed is not None:
        base_seed = opts.seed
    else:
        base_seed = torch.randint(0, 2**32, (1,)).item()

    prompts = opts.prompts

    total_images = len(prompts) * opts.num_images_per_prompt

    progress_bar = tqdm(total=total_images, desc="Generating images") 

    # Compute number of prompt batches
    num_prompt_batches = (len(prompts) + opts.batch_size - 1) // opts.batch_size

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(prompts))
        batch_prompts = prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate corresponding number of images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            # Prepare random seed
            seed = base_seed + idx  # Assign a different seed for each image
            idx += num_prompts_in_batch  # Update image index

            # Prepare input
            batch_size = num_prompts_in_batch
            x = get_noise(
                batch_size,
                opts.height,
                opts.width,
                device=torch_device,
                dtype=torch.bfloat16,
                seed=seed,
            )

            # Prepare prompts
            # batch_prompts is a list containing the prompts in the current batch
            inp = prepare(t5, clip, x, prompt=batch_prompts)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell"))
            
            # Denoising
            with torch.no_grad():
                if opts.test_FLOPs:
                    x = denoise_test_FLOPs(model, **inp, timesteps=timesteps, guidance=opts.guidance)
                else:
                    x = denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, model_kwargs=model_kwargs)
                    #x = search_denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, interval=opts.interval, max_order=opts.max_order, first_enhance=opts.first_enhance)

                # Decode latent variables
                x = unpack(x.float(), opts.height, opts.width)
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

            # Convert to PIL format and save
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x, "b c h w -> b h w c")

            for i in range(batch_size):
                img_array = x[i]
                img = Image.fromarray((127.5 * (img_array + 1.0)).cpu().byte().numpy())

                # Optional NSFW filtering
                if opts.use_nsfw_filter:
                    nsfw_result = nsfw_classifier(img)
                    nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0)
                else:
                    nsfw_score = 0.0  # If the filter is not enabled, assume safe

                if nsfw_score < NSFW_THRESHOLD:
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                    exif_data[ExifTags.Base.Model] = model_name
                    if opts.add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]
                    # Save image
                    fn = output_name.format(idx=idx - num_prompts_in_batch + i)
                    img.save(fn, exif=exif_data, quality=95, subsampling=0)
                else:
                    print(f"Generated image may contain inappropriate content, skipped.")

                progress_bar.update(1)

    progress_bar.close()


def read_prompts(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def app():
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the flux model.")
    parser.add_argument('--prompt_file', type=str, required=True, help='Path to the prompt text file.')
    parser.add_argument('--width', type=int, default=1360, help='Width of the generated image.')
    parser.add_argument('--height', type=int, default=768, help='Height of the generated image.')
    parser.add_argument('--num_steps', type=int, default=None, help='Number of sampling steps.')
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance value.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help='Number of images per prompt.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (prompt batching).')
    parser.add_argument('--model_name', type=str, default='flux-schnell', choices=['flux-dev', 'flux-schnell'], help='Model name.')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/samples', help='Directory to save images.')
    parser.add_argument('--add_sampling_metadata', action='store_true', help='Whether to add prompt metadata to images.')
    parser.add_argument('--use_nsfw_filter', action='store_true', help='Enable NSFW filter.')
    parser.add_argument('--test_FLOPs', action='store_true', help='Test inference computation cost.')
    parser.add_argument('--mode', type=str, default='Taylor',
                        choices=['embed', 'infer', 'Taylor-Cache', 'ToCa', 'Taylor', 'ClusCa'],
                        help='Pipeline mode: "embed" saves prompt embeddings to disk, '
                             '"infer" loads embeddings and runs denoising, '
                             'other values run the full pipeline with the specified cache strategy.')
    parser.add_argument('--embed_path', type=str, default=None,
                        help='Path to save/load prompt embeddings pickle file. '
                             'Used in embed/infer modes. Default: <output_dir>/embeddings.pkl')
    parser.add_argument('--cache_mode', type=str, default='Taylor',
                        choices=['Taylor-Cache', 'ToCa', 'Taylor', 'ClusCa'],
                        help='Cache strategy used in infer mode (default: Taylor).')
    parser.add_argument('--max_order', type=int, default=1, help='Max order of Taylor expansion.')
    parser.add_argument('--fresh_threshold', type=int, default=5, help='Fresh threshold.')
    
    parser.add_argument('--cluster_num', type=int, default=16, help='Number of clusters.')
    parser.add_argument('--cluster_method', type=str, default='kmeans', choices=['kmeans', 'kmeans++', 'random'], help='Clustering method.')
    parser.add_argument('--k', type=int, default=1, help='num of selected fresh tokens per cluster.')
    parser.add_argument('--propagation_ratio', type=float, default=0.005, help='Propagation ratio.')

    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)

    opts = SamplingOptions(
        prompts=prompts,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
        test_FLOPs=args.test_FLOPs,
        mode=args.mode,
    )

    model_kwargs = {
        'mode': args.mode, 
        'max_order': args.max_order,
        'fresh_threshold': args.fresh_threshold, 
        'propagation_ratio': args.propagation_ratio,
        'cluster_num': args.cluster_num,
        'cluster_method': args.cluster_method,
        'k': args.k,
        'embed_path': args.embed_path,
        'cache_mode': args.cache_mode,
    }

    main(opts, model_kwargs)


if __name__ == '__main__':
    app()
