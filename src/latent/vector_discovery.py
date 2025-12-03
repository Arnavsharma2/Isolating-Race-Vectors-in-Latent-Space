"""
Race vector extraction and discovery.

This module implements methods for finding linear directions in latent space
that correspond to racial/phenotypic attributes.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class RaceVectorExtractor:
    """
    Finds the "race" direction in the latent space.

    Ways to do it:
    - Supervised: Learn from pairs of images (e.g. same person, different race)
    - Unsupervised: PCA on a bunch of latents
    - Optimization: Refine it to minimize identity loss

    Example:
        >>> extractor = RaceVectorExtractor(method='supervised')
        >>> race_vector = extractor.extract_from_pairs(light_latents, dark_latents)
        >>> optimized_vector = extractor.optimize_vector(race_vector, identity_loss_fn)
    """

    def __init__(
        self,
        method: str = "supervised",
        device: str = "cuda",
    ):
        """
        Initialize vector extractor.

        Args:
            method: 'supervised', 'unsupervised', or 'pca'
            device: Device to run on
        """
        self.method = method
        self.device = device

    def extract_from_pairs(
        self,
        latents_a: List[torch.Tensor],
        latents_b: List[torch.Tensor],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Calculates the vector by comparing pairs of images.

        It just takes the average difference between the two groups.

        Args:
            latents_a: Latents for group A (e.g. light skin)
            latents_b: Latents for group B (e.g. dark skin)
            normalize: Whether to make the vector unit length

        Returns:
            The extracted race vector
        """
        if len(latents_a) != len(latents_b):
            raise ValueError("Must have same number of latents in each group")

        # Compute pairwise differences
        differences = []
        for lat_a, lat_b in zip(latents_a, latents_b):
            diff = lat_b - lat_a
            differences.append(diff)

        # Average all differences
        race_vector = torch.stack(differences).mean(dim=0)

        # Normalize
        if normalize:
            race_vector = race_vector / (race_vector.norm() + 1e-8)

        return race_vector

    def extract_from_groups(
        self,
        group_a_latents: List[torch.Tensor],
        group_b_latents: List[torch.Tensor],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract race vector from two groups (unpaired).

        Approach: difference of means.

        Args:
            group_a_latents: Latent codes for group A
            group_b_latents: Latent codes for group B
            normalize: Normalize the vector

        Returns:
            Race vector
        """
        # Compute means
        mean_a = torch.stack(group_a_latents).mean(dim=0)
        mean_b = torch.stack(group_b_latents).mean(dim=0)

        # Compute difference
        race_vector = mean_b - mean_a

        # Normalize
        if normalize:
            race_vector = race_vector / (race_vector.norm() + 1e-8)

        return race_vector

    def pca_based_extraction(
        self,
        latents: List[torch.Tensor],
        labels: np.ndarray,
        n_components: int = 10,
    ) -> torch.Tensor:
        """
        Extract race vector using PCA.

        Find principal component that correlates most with race labels.

        Args:
            latents: All latent codes
            labels: Binary labels (0 or 1) for race attribute
            n_components: Number of PCA components

        Returns:
            Race vector (principal component)
        """
        # Stack latents
        latents_stacked = torch.stack(latents).cpu().numpy()

        # Flatten to 2D
        orig_shape = latents_stacked.shape
        latents_flat = latents_stacked.reshape(len(latents), -1)

        # Apply PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(latents_flat)

        # Find component that best separates labels
        best_corr = 0
        best_idx = 0
        for i in range(n_components):
            corr = np.abs(np.corrcoef(components[:, i], labels)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_idx = i

        # Get best component
        race_vector_flat = pca.components_[best_idx]

        # Reshape back to latent shape
        race_vector = race_vector_flat.reshape(orig_shape[1:])
        race_vector = torch.from_numpy(race_vector).to(self.device)

        # Normalize
        race_vector = race_vector / (race_vector.norm() + 1e-8)

        return race_vector

    def optimize_vector(
        self,
        initial_vector: torch.Tensor,
        latents: List[torch.Tensor],
        identity_loss_fn,
        attribute_change_fn,
        num_iterations: int = 100,
        lr: float = 0.01,
        lambda_identity: float = 0.7,
        lambda_attribute: float = 0.3,
    ) -> torch.Tensor:
        """
        Refine vector to maximize disentanglement.

        Objective:
            L = λ_identity * L_identity + λ_attribute * (-L_attribute)

        Args:
            initial_vector: Starting vector
            latents: Training latent codes
            identity_loss_fn: Function to compute identity loss
            attribute_change_fn: Function to compute attribute change
            num_iterations: Optimization steps
            lr: Learning rate
            lambda_identity: Weight for identity preservation
            lambda_attribute: Weight for attribute change

        Returns:
            Optimized race vector
        """
        vector = initial_vector.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([vector], lr=lr)

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Sample batch of latents
            batch_size = min(8, len(latents))
            batch_indices = np.random.choice(len(latents), batch_size, replace=False)
            batch_latents = [latents[i] for i in batch_indices]

            # Apply vector
            modified_latents = [lat + vector for lat in batch_latents]

            # Compute losses
            identity_loss = identity_loss_fn(batch_latents, modified_latents)
            attribute_change = attribute_change_fn(batch_latents, modified_latents)

            # Combined loss (minimize identity loss, maximize attribute change)
            loss = lambda_identity * identity_loss - lambda_attribute * attribute_change

            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                print(
                    f"Iter {i+1}/{num_iterations}: "
                    f"Identity Loss: {identity_loss.item():.4f}, "
                    f"Attribute Change: {attribute_change.item():.4f}"
                )

        return vector.detach()

    def decompose_into_subvectors(
        self,
        race_vector: torch.Tensor,
        n_components: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose race vector into orthogonal subcomponents.

        Useful for finer-grained control (e.g., skin tone, hair, features).

        Args:
            race_vector: Full race vector
            n_components: Number of subcomponents

        Returns:
            Dictionary of subvectors
        """
        # Flatten
        vector_flat = race_vector.flatten().cpu().numpy()

        # Apply PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(vector_flat.reshape(1, -1))

        # Create subvector dict
        subvectors = {}
        labels = ["primary", "secondary", "tertiary"]
        for i in range(n_components):
            component = pca.components_[i].reshape(race_vector.shape)
            component = torch.from_numpy(component).to(self.device)
            subvectors[labels[i]] = component

        return subvectors

    def save_vector(self, vector: torch.Tensor, path: Path):
        """Save race vector to disk."""
        torch.save(vector.cpu(), path)
        print(f"✓ Saved race vector to {path}")

    def load_vector(self, path: Path) -> torch.Tensor:
        """Load race vector from disk."""
        vector = torch.load(path, map_location=self.device)
        print(f"✓ Loaded race vector from {path}")
        return vector


class VectorAnalyzer:
    """
    Analyze properties of discovered race vectors.

    Tools for understanding:
    - Vector magnitude
    - Activation patterns
    - Correlation with known attributes
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def compute_magnitude(self, vector: torch.Tensor) -> float:
        """Compute L2 norm of vector."""
        return vector.norm().item()

    def analyze_spatial_pattern(
        self,
        vector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze spatial activation pattern.

        Returns:
            Dict with statistics per channel and spatial location
        """
        # Per-channel magnitude
        per_channel = vector.pow(2).sum(dim=(1, 2))

        # Spatial heatmap (average across channels)
        spatial_heatmap = vector.pow(2).mean(dim=0)

        return {
            "per_channel_magnitude": per_channel,
            "spatial_heatmap": spatial_heatmap,
            "total_magnitude": self.compute_magnitude(vector),
        }

    def compute_orthogonality(
        self,
        vector1: torch.Tensor,
        vector2: torch.Tensor,
    ) -> float:
        """
        Compute orthogonality between two vectors.

        Returns:
            Cosine similarity (0 = orthogonal, 1 = parallel)
        """
        v1_flat = vector1.flatten()
        v2_flat = vector2.flatten()

        cosine_sim = torch.nn.functional.cosine_similarity(
            v1_flat.unsqueeze(0),
            v2_flat.unsqueeze(0),
        )

        return cosine_sim.item()
