"""
datasets.py
-----------
A dedicated module containing different benchmark datasets for testing and 
demonstrating the OPTICS (Ordering Points To Identify the Clustering Structure) algorithm.

All datasets are generated from scratch using pure mathematical formulas and 
random sampling (No scikit-learn or black-box libraries).

Prepared for college presentation to demonstrate how OPTICS handles:
1. Varying density clusters (OPTICS's primary strength over DBSCAN)
2. Non-linear / crescent shapes (Two Moons)
3. Concentric rings / non-convex shapes
4. Separated spherical clusters with noise
5. Hierarchical / nested density clusters
"""

import numpy as np


def generate_varying_density(seed=42):
    """
    Dataset 1: Clusters with significantly DIFFERENT densities + noise.
    
    Why this matters for your presentation:
    This is the classic scenario where DBSCAN fails because a single global epsilon
    cannot capture both dense and sparse clusters. OPTICS succeeds because its reachability
    plot displays valleys of different depths (dense = deep valley, sparse = shallower valley).
    """
    np.random.seed(seed)
    
    # Dense cluster (small spread, high concentration)
    c1 = np.random.normal(loc=[2.0, 2.0], scale=0.25, size=(70, 2))
    
    # Medium density cluster
    c2 = np.random.normal(loc=[6.5, 6.5], scale=0.65, size=(80, 2))
    
    # Sparse density cluster (wide spread, lower concentration)
    c3 = np.random.normal(loc=[3.0, 7.5], scale=1.1, size=(90, 2))
    
    # Random background noise
    noise = np.random.uniform(low=0.0, high=10.0, size=(30, 2))
    
    data = np.vstack([c1, c2, c3, noise])
    
    description = (
        "Varying Density Clusters: 3 clusters with different spreads + noise.\n"
        "Key insight: Deep valley = high density; shallow valley = lower density.\n"
        "DBSCAN struggles with this, but OPTICS easily captures both!"
    )
    suggested_params = {"eps": 1.8, "min_pts": 6, "eps_prime": 0.9}
    return data, description, suggested_params


def generate_two_moons(seed=42):
    """
    Dataset 2: Two interleaving crescent shapes (moons) + noise.
    
    Why this matters for your presentation:
    Demonstrates that density-based clustering finds arbitrarily shaped, 
    non-convex clusters where centroid-based algorithms (like K-Means) completely fail.
    """
    np.random.seed(seed)
    n_samples_per_moon = 80
    
    # Upper moon
    theta1 = np.linspace(0, np.pi, n_samples_per_moon)
    x1 = np.cos(theta1) + np.random.normal(0, 0.08, n_samples_per_moon)
    y1 = np.sin(theta1) + np.random.normal(0, 0.08, n_samples_per_moon)
    moon1 = np.column_stack([x1, y1])
    
    # Lower moon
    theta2 = np.linspace(0, np.pi, n_samples_per_moon)
    x2 = 1.0 - np.cos(theta2) + np.random.normal(0, 0.08, n_samples_per_moon)
    y2 = 0.5 - np.sin(theta2) - 0.2 + np.random.normal(0, 0.08, n_samples_per_moon)
    moon2 = np.column_stack([x2, y2])
    
    # Scattered noise
    noise = np.random.uniform(low=[-1.2, -1.0], high=[2.2, 1.3], size=(25, 2))
    
    data = np.vstack([moon1, moon2, noise])
    
    description = (
        "Two Moons: Interleaving non-linear crescent shapes.\n"
        "Key insight: OPTICS follows density connectivity along curved manifolds\n"
        "producing 2 distinct valleys corresponding to the two crescents."
    )
    suggested_params = {"eps": 0.45, "min_pts": 5, "eps_prime": 0.28}
    return data, description, suggested_params


def generate_concentric_circles(seed=42):
    """
    Dataset 3: Concentric rings (inner core and outer ring) + noise.
    
    Why this matters for your presentation:
    Highlights topological/ring-shaped clustering capability.
    """
    np.random.seed(seed)
    
    # Inner circular cluster
    r_inner = np.random.uniform(0.0, 0.8, 60)
    theta_inner = np.random.uniform(0, 2 * np.pi, 60)
    inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    
    # Outer ring cluster
    r_outer = np.random.normal(2.5, 0.15, 120)
    theta_outer = np.random.uniform(0, 2 * np.pi, 120)
    outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
    
    # Random noise points
    noise = np.random.uniform(low=-3.2, high=3.2, size=(25, 2))
    
    data = np.vstack([inner, outer, noise])
    
    description = (
        "Concentric Circles: Inner core and outer surrounding ring.\n"
        "Key insight: Density-connectivity identifies the ring as one continuous cluster\n"
        "and the inner circle as another, separated by high reachability spikes."
    )
    suggested_params = {"eps": 0.8, "min_pts": 6, "eps_prime": 0.45}
    return data, description, suggested_params


def generate_separated_blobs(seed=42):
    """
    Dataset 4: 3 Well-separated Gaussian blobs with uniform background noise.
    
    Why this matters for your presentation:
    Clear baseline showing clean, separated valleys and noise spikes.
    Very easy for the audience to intuitively understand the reachability graph.
    """
    np.random.seed(seed)
    
    b1 = np.random.normal(loc=[2.0, 2.0], scale=0.45, size=(60, 2))
    b2 = np.random.normal(loc=[7.5, 2.5], scale=0.45, size=(60, 2))
    b3 = np.random.normal(loc=[5.0, 7.5], scale=0.45, size=(60, 2))
    noise = np.random.uniform(low=0.0, high=10.0, size=(30, 2))
    
    data = np.vstack([b1, b2, b3, noise])
    
    description = (
        "Separated Blobs with Noise: 3 clear clusters and ambient noise.\n"
        "Key insight: 3 wide, clear valleys separated by tall reachability peaks.\n"
        "The tall peaks represent the transition across empty space or noise."
    )
    suggested_params = {"eps": 2.0, "min_pts": 5, "eps_prime": 1.0}
    return data, description, suggested_params


def generate_nested_clusters(seed=42):
    """
    Dataset 5: Hierarchical / Nested Clusters (Dense core inside a diffuse cluster).
    
    Why this matters for your presentation:
    OPTICS was specifically designed to capture hierarchical clustering structure.
    In the reachability plot, you will see a 'sub-valley' inside a larger valley!
    """
    np.random.seed(seed)
    
    # Dense inner core
    core = np.random.normal(loc=[4.0, 4.0], scale=0.3, size=(70, 2))
    
    # Diffuse surrounding halo
    halo = np.random.normal(loc=[4.0, 4.0], scale=1.4, size=(90, 2))
    
    # Separate distinct cluster
    c2 = np.random.normal(loc=[8.5, 8.5], scale=0.4, size=(50, 2))
    
    # Noise
    noise = np.random.uniform(low=1.0, high=10.0, size=(20, 2))
    
    data = np.vstack([core, halo, c2, noise])
    
    description = (
        "Nested / Hierarchical Clusters: A dense core inside a diffuse halo.\n"
        "Key insight: Shows hierarchical clustering! A deeper valley sits within\n"
        "a broader valley, proving OPTICS's hierarchical detection power."
    )
    suggested_params = {"eps": 1.5, "min_pts": 7, "eps_prime": 0.65}
    return data, description, suggested_params


def load_csv_dataset(filepath):
    """
    Loads custom numerical 2D dataset from a CSV file.
    Supports CSV with or without headers.
    """
    try:
        # Read only the first two numeric columns; later columns may contain labels.
        data = np.genfromtxt(filepath, delimiter=',', usecols=(0, 1), comments='#')
        data = np.atleast_2d(data)
        # A text header becomes a NaN row, while headerless files remain unchanged.
        data = data[~np.isnan(data).any(axis=1)]

        if data.shape[0] == 0:
            raise ValueError("CSV does not contain numeric data in its first two columns.")

        description = f"Custom CSV Dataset loaded from '{filepath}' ({len(data)} points)."
        suggested_params = {"eps": 0.4, "min_pts": 5, "eps_prime": 0.36}
        return data, description, suggested_params
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file '{filepath}': {e}")


# Registry of available datasets
DATASET_REGISTRY = {
    "1": {
        "name": "Varying Density Clusters (Recommended for Presentation)",
        "generator": generate_varying_density
    },
    "2": {
        "name": "Two Moons (Non-linear Crescent Shapes)",
        "generator": generate_two_moons
    },
    "3": {
        "name": "Concentric Circles (Ring Structure)",
        "generator": generate_concentric_circles
    },
    "4": {
        "name": "Separated Blobs with Noise (Classic Benchmark)",
        "generator": generate_separated_blobs
    },
    "5": {
        "name": "Nested / Hierarchical Clusters",
        "generator": generate_nested_clusters
    }
}


def get_available_datasets():
    """Returns the list of dataset options for user selection."""
    return DATASET_REGISTRY
