import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    from src.models.stable_diffusion import StableDiffusionWrapper
    from src.latent.vector_discovery import RaceVectorExtractor
    from src.latent.manipulator import LatentManipulator
    from src.metrics.evaluator import CounterfactualEvaluator
    from src.visualization.grid_generator import CounterfactualGridGenerator
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def main():
    print("Starting verification...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model
    try:
        model = StableDiffusionWrapper(
            device=device,
            dtype=torch.float32, # Use float32 for CPU/compatibility
            enable_xformers=False,
        )
        print("Model loaded.")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    # 2. Generate Test Images (Fast)
    print("Generating test images...")
    try:
        img_light, lat_light = model.generate_from_prompt("light skin person", seed=42, num_inference_steps=1)
        img_dark, lat_dark = model.generate_from_prompt("dark skin person", seed=43, num_inference_steps=1)
        print("Images generated.")
    except Exception as e:
        print(f"Generation failed: {e}")
        return

    # 3. Extract Vector
    print("Extracting vector...")
    try:
        extractor = RaceVectorExtractor(device=device)
        race_vector = extractor.extract_from_pairs([lat_light], [lat_dark], normalize=True)
        print(f"Vector extracted. Norm: {race_vector.norm().item()}")
    except Exception as e:
        print(f"Vector extraction failed: {e}")
        return

    # 4. Generate Counterfactuals
    print("Generating counterfactuals...")
    try:
        manipulator = LatentManipulator(device=device)
        cf_latents = manipulator.generate_counterfactuals(lat_light, race_vector, alphas=[-1.0, 1.0])
        cf_images = [model.decode_latent(l) for l in cf_latents]
        print("Counterfactuals generated.")
    except Exception as e:
        print(f"Counterfactual generation failed: {e}")
        return

    # 5. Evaluate (Mocking or minimal run)
    print("Evaluating...")
    try:
        evaluator = CounterfactualEvaluator(device=device)
        # Note: This might fail if models need downloading, so we wrap carefully
        # For now just check if class instantiates
        print("Evaluator instantiated.")
    except Exception as e:
        print(f"Evaluation setup failed: {e}")
        return

    print("Verification successful!")

if __name__ == "__main__":
    main()
