#!/usr/bin/env python3
"""
Race Vector Extraction and Counterfactual Generation

This script extracts a race vector from portrait photos and generates
counterfactual images showing skin tone transformations.

Usage:
    python3 run_race_vector_extraction.py

Requirements:
    - Photos in data/photos/light_skin/ and data/photos/dark_skin/
    - At least 3 photos in each directory
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.stable_diffusion import StableDiffusionWrapper
from src.latent.vector_discovery import RaceVectorExtractor, VectorAnalyzer
from src.metrics.evaluator import CounterfactualEvaluator
from src.visualization.grid_generator import CounterfactualGridGenerator


class RaceVectorPipeline:
    """End-to-end pipeline for race vector extraction and evaluation"""

    def __init__(self, device=None, output_dir="experiments/results"):
        """Initialize pipeline"""
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache directory for generated assets to reuse across runs
        self.cache_dir = Path("data/generated")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("RACE VECTOR EXTRACTION PIPELINE")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print()

    def load_model(self):
        """Load Stable Diffusion model"""
        print("STEP 1: Loading Stable Diffusion model...")
        print("-" * 70)

        self.model = StableDiffusionWrapper(
            device=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            enable_xformers=True,
        )
        print("Model loaded.\n")

    def load_photos(self, light_dir="data/photos/light_skin",
                   dark_dir="data/photos/dark_skin", max_photos=10):
        """Load photos from directories"""
        print("STEP 2: Loading photos...")
        print("-" * 70)

        light_path = Path(light_dir)
        dark_path = Path(dark_dir)

        # Find all image files
        light_files = (list(light_path.glob("*.jpg")) +
                      list(light_path.glob("*.jpeg")) +
                      list(light_path.glob("*.png")))
        dark_files = (list(dark_path.glob("*.jpg")) +
                     list(dark_path.glob("*.jpeg")) +
                     list(dark_path.glob("*.png")))

        print(f"Found {len(light_files)} light skin photos")
        print(f"Found {len(dark_files)} dark skin photos")

        if len(light_files) == 0 or len(dark_files) == 0:
            print("\nERROR: No photos found!")
            print(f"   Light skin: {light_path.absolute()}")
            print(f"   Dark skin: {dark_path.absolute()}")
            print("\nRun: python3 get_sample_photos.py")
            sys.exit(1)

        # Load and encode
        self.light_images = []
        self.light_latents = []

        for img_path in sorted(light_files)[:max_photos]:
            print(f"  Encoding {img_path.name}...")
            img = Image.open(img_path).convert("RGB")
            img = img.resize((512, 512), Image.LANCZOS)
            latent = self.model.encode_image(img)
            self.light_images.append(img)
            self.light_latents.append(latent)

        self.dark_images = []
        self.dark_latents = []

        for img_path in sorted(dark_files)[:max_photos]:
            print(f"  Encoding {img_path.name}...")
            img = Image.open(img_path).convert("RGB")
            img = img.resize((512, 512), Image.LANCZOS)
            latent = self.model.encode_image(img)
            self.dark_images.append(img)
            self.dark_latents.append(latent)

        print(f"Loaded {len(self.light_images)} light + {len(self.dark_images)} dark photos\n")

    def check_photo_quality(self):
        """Check if photos have sufficient contrast"""
        print("STEP 3: Checking photo quality...")
        print("-" * 70)

        def get_avg_brightness(images):
            """Get average brightness of center region"""
            brightnesses = []
            for img in images:
                img_array = np.array(img)
                h, w = img_array.shape[:2]
                center = img_array[h//4:3*h//4, w//4:3*w//4]
                brightnesses.append(center.mean())
            return np.mean(brightnesses)

        light_brightness = get_avg_brightness(self.light_images)
        dark_brightness = get_avg_brightness(self.dark_images)
        diff = abs(light_brightness - dark_brightness)

        print(f"Light skin avg brightness: {light_brightness:.1f}")
        print(f"Dark skin avg brightness:  {dark_brightness:.1f}")
        print(f"Difference: {diff:.1f}")

        if diff < 20:
            print("\nWARNING: Photos are too similar.")
            print("   Race vector may be weak. Recommend difference > 40")
        elif diff < 40:
            print("\nWARNING: Small difference (recommend > 40)")
        else:
            print(f"\nGood contrast. ({diff:.1f})")
        print()

    def extract_race_vector(self, radius=0.8, edge_weight=0.0):
        """Extract race vector with spatial masking"""
        print("STEP 4: Extracting race vector...")
        print("-" * 70)

        self.extractor = RaceVectorExtractor(device=self.device)

        # Get latent shape
        latent_shape = self.light_latents[0].shape
        if len(latent_shape) == 4:
            h, w = latent_shape[-2], latent_shape[-1]
        else:
            h, w = latent_shape[-2], latent_shape[-1]

        # Create spatial mask
        spatial_mask = self.extractor.create_center_mask(
            height=h,
            width=w,
            center_weight=1.0,
            edge_weight=edge_weight,
            falloff='gaussian',
            radius=radius,
        )

        print(f"Created spatial mask (radius={radius}, edge_weight={edge_weight})")
        print(f"  Center: {spatial_mask[h//2, w//2].item():.4f}")
        print(f"  Edge: {spatial_mask[0, 0].item():.6f}")

        self.race_vector = self.extractor.extract_from_pairs(
            self.light_latents,
            self.dark_latents,
            normalize=False,
            spatial_mask=spatial_mask,
        )

        print(f"\nRace vector extracted.")
        print(f"  Shape: {self.race_vector.shape}")
        print(f"  Norm: {self.race_vector.norm().item():.4f}\n")

        # Save visualization
        self.visualize_mask_and_vector(spatial_mask)

        # ---------------------------------------------------------
        # OPTIMIZATION STEP
        # ---------------------------------------------------------
        print("\nSTEP 4b: Optimizing vector for identity preservation...")
        print("-" * 70)
        
        try:
            from facenet_pytorch import InceptionResnetV1
            import torch.nn.functional as F
            
            # Load FaceNet for identity loss
            # We need a differentiable loss, so we use the model directly
            facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            # Freeze FaceNet
            for param in facenet.parameters():
                param.requires_grad = False
                
            def differentiable_identity_loss(latents_batch, modified_latents_batch):
                """Compute identity loss (cosine distance) in a differentiable way."""
                loss_sum = 0
                for orig, mod in zip(latents_batch, modified_latents_batch):
                    # Decode (unsqueezing to add batch dim if needed)
                    if orig.dim() == 3: orig = orig.unsqueeze(0)
                    if mod.dim() == 3: mod = mod.unsqueeze(0)
                    
                    # Manual decode to ensure gradients flow
                    orig_scaled = orig / self.model.vae.config.scaling_factor
                    mod_scaled = mod / self.model.vae.config.scaling_factor
                    
                    orig_img_tensor = self.model.vae.decode(orig_scaled).sample
                    mod_img_tensor = self.model.vae.decode(mod_scaled).sample
                    
                    # Resize for FaceNet (160x160)
                    orig_face = F.interpolate(orig_img_tensor, size=(160, 160), mode='bilinear')
                    mod_face = F.interpolate(mod_img_tensor, size=(160, 160), mode='bilinear')
                    
                    # Get embeddings
                    emb_orig = facenet(orig_face)
                    emb_mod = facenet(mod_face)
                    
                    loss = 1 - F.cosine_similarity(emb_orig, emb_mod).mean()
                    loss_sum += loss
                    
                return loss_sum / len(latents_batch)

            def attribute_change_loss(latents_batch, modified_latents_batch):
                diff = torch.stack(modified_latents_batch) - torch.stack(latents_batch)
                return diff.norm(p=2)

            print("  Optimizing vector... (this may take a few minutes)")

            # Use a subset of latents for optimization to save time/memory
            train_latents = self.light_latents[:3] + self.dark_latents[:3]

            self.race_vector = self.extractor.optimize_vector(
                initial_vector=self.race_vector,
                latents=train_latents,
                identity_loss_fn=differentiable_identity_loss,
                attribute_change_fn=attribute_change_loss,
                num_iterations=50,
                lr=0.01,
                lambda_identity=0.7,
                lambda_attribute=0.3,
            )
            
            print(f"Optimized Vector Norm: {self.race_vector.norm().item():.4f}")
            
            # Re-visualize optimized vector
            self.visualize_mask_and_vector(spatial_mask)
            
        except ImportError:
            print("WARNING: facenet-pytorch not found. Skipping optimization.")
        except Exception as e:
            print(f"WARNING: Optimization failed: {e}")
            print("  Using raw vector instead.")


    def visualize_mask_and_vector(self, spatial_mask):
        """Visualize spatial mask and vector activation"""
        analyzer = VectorAnalyzer(device=self.device)
        analysis = analyzer.analyze_spatial_pattern(self.race_vector)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Spatial mask
        im0 = axes[0].imshow(spatial_mask.cpu().numpy(), cmap='hot')
        axes[0].set_title('Spatial Mask\n(High = face, Low = background)')
        axes[0].set_xlabel('Width')
        axes[0].set_ylabel('Height')
        plt.colorbar(im0, ax=axes[0], label='Weight')

        # Vector activation
        im1 = axes[1].imshow(analysis['spatial_heatmap'].cpu().numpy(), cmap='hot')
        axes[1].set_title('Race Vector Activation\n(After masking)')
        axes[1].set_xlabel('Width')
        axes[1].set_ylabel('Height')
        plt.colorbar(im1, ax=axes[1], label='Magnitude')

        plt.tight_layout()
        mask_path = self.output_dir / "spatial_mask_and_vector.png"
        plt.savefig(mask_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization: {mask_path}")

    def generate_base_image(self, prompt=None, seed=999):
        """Generate base image and save the prompt/seed for counterfactual steering."""
        print("STEP 5: Generating base image...")
        print("-" * 70)

        if prompt is None:
            prompt = "portrait photo of a person, professional headshot, neutral background, high quality"

        # Store for reuse during steered counterfactual generation
        self.generation_prompt = prompt
        self.generation_seed = seed
        self.generation_negative_prompt = "multiple people, accessories, jewelry, glasses"

        cache_path = self.cache_dir / "base_image.png"

        # Check cache first
        if cache_path.exists():
            print(f"Found cached base image: {cache_path}")
            print("  Loading from cache (skipping generation)...")
            try:
                self.base_image = Image.open(cache_path).convert("RGB")
                self.base_image = self.base_image.resize((512, 512), Image.LANCZOS)
                base_path = self.output_dir / "base_image.png"
                self.base_image.save(base_path)
                print(f"Base image loaded and copied to: {base_path}\n")
                return
            except Exception as e:
                print(f"WARNING: Failed to load cache: {e}")
                print("  Falling back to generation...")

        print(f"Prompt: {prompt}")
        print(f"Seed: {seed}")

        self.base_image, _ = self.model.generate_from_prompt(
            prompt,
            negative_prompt=self.generation_negative_prompt,
            seed=seed,
            num_inference_steps=50,
            guidance_scale=7.5,
        )

        print(f"  Saving to cache: {cache_path}")
        self.base_image.save(cache_path)

        base_path = self.output_dir / "base_image.png"
        self.base_image.save(base_path)
        print(f"Base image saved: {base_path}\n")

    def generate_counterfactuals(self, alphas=None):
        """
        Generate counterfactuals using steered denoising.

        Each counterfactual is generated fresh from the same seed as the base
        image, but with the race vector injected at every denoising step.
        This keeps the generation in-distribution (no "misty" VAE artifacts)
        while still moving along the race direction in latent space.

        Alpha tuning guide:
          - Start with the defaults below and widen if the effect is too subtle.
          - Values beyond ±6 often produce identity-breaking changes.
        """
        print("STEP 6: Generating counterfactuals (steered denoising)...")
        print("-" * 70)

        if alphas is None:
            alphas = [-4, -2, 0, 2, 4]

        print(f"Alpha values: {alphas}")
        print("(Negative = lighter skin, Positive = darker skin)")
        print("Using same seed as base image for consistent identity.\n")

        self.alphas = alphas
        self.counterfactual_images = []

        for alpha in alphas:
            print(f"  Generating α = {alpha:+.1f}...", end=" ", flush=True)

            if abs(alpha) < 0.01:
                # α=0 is the unsteered base image — just re-use it
                self.counterfactual_images.append(self.base_image)
                print("(base image, skipped)")
                continue

            img, _ = self.model.generate_steered(
                prompt=self.generation_prompt,
                race_vector=self.race_vector,
                alpha=alpha,
                seed=self.generation_seed,
                negative_prompt=self.generation_negative_prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
            )
            self.counterfactual_images.append(img)
            print("done")

        print(f"\nGenerated {len(self.counterfactual_images)} counterfactuals\n")
        self.visualize_counterfactuals()

    def visualize_counterfactuals(self):
        """Visualize counterfactuals in a grid"""
        fig, axes = plt.subplots(1, len(self.alphas), figsize=(20, 4))

        for i, (img, alpha) in enumerate(zip(self.counterfactual_images, self.alphas)):
            axes[i].imshow(img)
            axes[i].set_title(f"α = {alpha:.1f}")
            axes[i].axis('off')

        plt.tight_layout()
        cf_path = self.output_dir / "counterfactuals.png"
        plt.savefig(cf_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved counterfactuals: {cf_path}")

    def evaluate_counterfactuals(self):
        """Evaluate identity preservation and disentanglement"""
        print("\nSTEP 7: Evaluating counterfactuals...")
        print("-" * 70)

        evaluator = CounterfactualEvaluator(device=self.device)

        self.results = []
        for cf_image, alpha in zip(self.counterfactual_images, self.alphas):
            if abs(alpha) < 0.01:  # Skip α=0
                continue

            print(f"\nEvaluating α = {alpha:.1f}")
            result = evaluator.evaluate_pair(
                self.base_image,
                cf_image,
                verbose=True,
            )
            self.results.append(result)

        print("\nEvaluation complete.\n")

    def create_final_grid(self):
        """Create final visualization grid with metrics"""
        print("STEP 8: Creating final visualization grid...")
        print("-" * 70)

        generator = CounterfactualGridGenerator()

        # Exclude α=0
        cf_images_no_orig = [img for img, a in zip(self.counterfactual_images, self.alphas)
                             if abs(a) >= 0.01]
        labels = [f"α={a:.1f}" for a in self.alphas if abs(a) >= 0.01]
        metrics_list = [r.to_dict() for r in self.results]

        grid = generator.generate_grid(
            self.base_image,
            cf_images_no_orig,
            labels=labels,
            metrics=metrics_list,
            title="Race Vector Counterfactual Generation",
        )

        grid_path = self.output_dir / "final_grid.png"
        grid.save(grid_path)
        print(f"Final grid saved: {grid_path}\n")

    def print_summary(self):
        """Print summary statistics"""
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nPhotos used:")
        print(f"  Light skin: {len(self.light_images)}")
        print(f"  Dark skin: {len(self.dark_images)}")
        print(f"\nRace vector:")
        print(f"  Norm: {self.race_vector.norm().item():.4f}")
        print(f"\nCounterfactuals generated: {len(self.counterfactual_images)}")
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"  - base_image.png")
        print(f"  - counterfactuals.png")
        print(f"  - spatial_mask_and_vector.png")
        print(f"  - final_grid.png")
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)


def main():
    """Run the complete pipeline"""
    pipeline = RaceVectorPipeline()

    try:
        # Step 1: Load model
        pipeline.load_model()

        # Step 2: Load photos
        pipeline.load_photos()

        # Step 3: Check quality
        pipeline.check_photo_quality()

        # Step 4: Extract race vector
        # Widen mask (0.8 -> 1.0) and soften edges (0.0 -> 0.3) to cover full face and blend better
        pipeline.extract_race_vector(radius=1.0, edge_weight=0.3)

        # Step 5: Generate base image
        pipeline.generate_base_image()

        # Step 6: Generate counterfactuals (steered denoising, default alphas [-4,-2,0,2,4])
        pipeline.generate_counterfactuals()

        # Step 7: Evaluate
        pipeline.evaluate_counterfactuals()

        # Step 8: Create final grid
        pipeline.create_final_grid()

        # Summary
        pipeline.print_summary()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
