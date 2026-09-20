"""
main.py
-------
OPTICS (Ordering Points To Identify the Clustering Structure) Algorithm
Implemented 100% from scratch without any built-in clustering or distance libraries.

Designed for college presentation with:
- Interactive parameter input (epsilon, minpoints, threshold eps_prime)
- Reachability Plot showing valleys (Y-axis: Reachability score, X-axis: Ordering of points)
- Side-by-side 2D Cluster visualization with matching colors
- Integration with datasets.py for multiple data distributions
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import datasets


# =====================================================================
# 1. CORE DISTANCE & NEIGHBORHOOD FUNCTIONS (100% From Scratch)
# =====================================================================

def euclidean_distance(point1, point2):
    """
    Computes Euclidean distance between two points from first principles:
    d(p, q) = sqrt( sum( (p_i - q_i)^2 ) )
    """
    diff = point1 - point2
    return np.sqrt(np.sum(diff * diff))


def get_neighbors(data, point_idx, eps):
    """
    Finds indices of all points within the epsilon neighborhood of point_idx:
    N_eps(p) = { q in Data | dist(p, q) <= eps }
    """
    neighbors = []
    p = data[point_idx]
    for i in range(len(data)):
        if euclidean_distance(p, data[i]) <= eps:
            neighbors.append(i)
    return neighbors


def get_core_distance(data, point_idx, eps, min_pts):
    """
    Computes the Core Distance of a point:
    - If |N_eps(p)| < min_pts, core_dist is undefined (infinity).
    - Otherwise, core_dist is the distance to the min_pts-th nearest neighbor.
    
    Explanation for presentation:
    Core distance represents the minimum radius needed around point p 
    to be considered a core (dense) point with at least min_pts points.
    """
    neighbors = get_neighbors(data, point_idx, eps)
    if len(neighbors) < min_pts:
        return float('inf')
    
    # Calculate pairwise distances to all points in the neighborhood
    p = data[point_idx]
    distances = [euclidean_distance(p, data[n]) for n in neighbors]
    distances.sort()
    
    # Note: distances[0] is 0 (distance to self).
    # The min_pts-th nearest neighbor is at index min_pts - 1.
    return distances[min_pts - 1]


# =====================================================================
# 2. OPTICS ALGORITHM (Ordering Points To Identify Clustering Structure)
# =====================================================================

def run_optics(data, eps, min_pts):
    """
    Runs the pure from-scratch OPTICS algorithm.
    
    Returns:
    --------
    ordered_list : list of int
        Indices of points in the order they were processed (cluster-ordering).
    reachability : list of float
        Reachability distance for each point in the dataset.
    core_distances : list of float
        Core distance for each point in the dataset.
    """
    n = len(data)
    processed = [False] * n
    reachability = [float('inf')] * n
    core_distances = [float('inf')] * n
    ordered_list = []

    def update_seeds(neighbors, current_idx, seeds):
        """
        Updates the priority queue (seeds) with reachability distances
        relative to current_idx.
        
        Reachability distance of point o relative to core point p:
        r(o, p) = max(core_distance(p), dist(p, o))
        """
        c_dist = core_distances[current_idx]
        p_vec = data[current_idx]
        
        for n_idx in neighbors:
            if not processed[n_idx]:
                actual_dist = euclidean_distance(p_vec, data[n_idx])
                new_reach_dist = max(c_dist, actual_dist)
                
                if reachability[n_idx] == float('inf'):
                    reachability[n_idx] = new_reach_dist
                    seeds[n_idx] = new_reach_dist
                elif new_reach_dist < reachability[n_idx]:
                    # Update with the smaller reachability distance found
                    reachability[n_idx] = new_reach_dist
                    seeds[n_idx] = new_reach_dist

    # Main OPTICS loop: Iterate through all unvisited points
    for i in range(n):
        if not processed[i]:
            neighbors = get_neighbors(data, i, eps)
            processed[i] = True
            ordered_list.append(i)
            core_distances[i] = get_core_distance(data, i, eps, min_pts)

            # If point i is a core point, explore its density-connected cluster
            if core_distances[i] != float('inf'):
                seeds = {}  # Priority queue: {point_index: reachability_distance}
                update_seeds(neighbors, i, seeds)
                
                # Continuously pop the point with the smallest reachability distance
                while seeds:
                    # Find candidate with minimum reachability distance
                    q_idx = min(seeds, key=seeds.get)
                    del seeds[q_idx]

                    processed[q_idx] = True
                    ordered_list.append(q_idx)

                    q_core_dist = get_core_distance(data, q_idx, eps, min_pts)
                    core_distances[q_idx] = q_core_dist

                    if q_core_dist != float('inf'):
                        q_neighbors = get_neighbors(data, q_idx, eps)
                        update_seeds(q_neighbors, q_idx, seeds)

    return ordered_list, reachability, core_distances


# =====================================================================
# 3. CLUSTER EXTRACTION FROM REACHABILITY PLOT (From Scratch)
# =====================================================================

def extract_clusters_dbscan_style(ordered_list, reachability, core_distances, eps_prime):
    """
    Extracts clusters using the threshold cut eps_prime (eps_prime <= eps) 
    directly from the reachability graph, as defined in the original OPTICS paper:
    
    - A reachability score > eps_prime represents a peak / valley boundary.
    - If the point's core distance <= eps_prime, it starts a NEW cluster.
    - Subsequent points with reachability <= eps_prime join the current cluster.
    - Points with reachability > eps_prime that are not core points are NOISE (-1).
    """
    labels = [-1] * len(ordered_list)
    current_cluster = -1
    
    for idx in ordered_list:
        if reachability[idx] > eps_prime:
            # Start of a potential new cluster if this point is dense enough at eps_prime
            if core_distances[idx] <= eps_prime:
                current_cluster += 1
                labels[idx] = current_cluster
            else:
                labels[idx] = -1  # Noise / outlier
        else:
            # Reachability <= eps_prime (inside a valley)
            if current_cluster >= 0:
                labels[idx] = current_cluster
            else:
                # If no cluster has started yet, check if it can start one
                if core_distances[idx] <= eps_prime:
                    current_cluster += 1
                    labels[idx] = current_cluster
                else:
                    labels[idx] = -1
                    
    return labels


# =====================================================================
# 4. PRESENTATION-READY VISUALIZATION (Dual-Panel Valley & Cluster Plot)
# =====================================================================

def plot_optics_presentation(data, ordered_list, reachability, labels, eps_prime, dataset_title=""):
    """
    Generates a dual-panel visualization specifically designed for a college presentation:
    
    Panel 1 (Left): Reachability Plot
      - X-axis: Ordering of points (processing order 0, 1, ..., N-1)
      - Y-axis: Reachability score / distance
      - Valleys are colored according to extracted clusters
      - Threshold cut line (eps_prime) showing cluster boundaries
      - Annotations highlighting valleys (clusters) and peaks (noise/separation)
      
    Panel 2 (Right): 2D Cluster Space
      - Points in 2D space colored identically to the reachability valleys
      - Noise points marked clearly as gray/black crosses ('x')
    """
    n_points = len(ordered_list)
    
    # Reorder reachability and labels according to the OPTICS ordering
    ordered_reachability = [reachability[i] for i in ordered_list]
    ordered_labels = [labels[i] for i in ordered_list]
    
    # Cap infinite reachability values for plotting
    finite_reaches = [r for r in ordered_reachability if r != float('inf')]
    max_finite = max(finite_reaches) if finite_reaches else 1.0
    cap_val = max_finite * 1.25
    plot_reachabilities = [r if r != float('inf') else cap_val for r in ordered_reachability]

    # Color palette
    unique_labels = sorted(list(set(labels)))
    cluster_ids = [l for l in unique_labels if l != -1]
    n_clusters = len(cluster_ids)
    
    # Generate distinct vibrant colors for clusters
    cmap = plt.get_cmap('tab10', max(10, n_clusters))
    cluster_colors = {}
    for i, c_id in enumerate(cluster_ids):
        cluster_colors[c_id] = cmap(i % 10)
    cluster_colors[-1] = (0.7, 0.7, 0.7, 0.7)  # Light gray for noise

    # Create figure with 2 subplots
    fig, (ax_reach, ax_space) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor('#fdfdfd')

    # -------------------------------------------------------------
    # SUBPLOT 1: REACHABILITY PLOT (VALLEYS & PEAKS)
    # -------------------------------------------------------------
    bar_colors = [cluster_colors[l] for l in ordered_labels]
    ax_reach.bar(range(n_points), plot_reachabilities, width=1.0, color=bar_colors, edgecolor='none', alpha=0.9)
    
    # Plot the threshold line eps_prime
    ax_reach.axhline(y=eps_prime, color='red', linestyle='--', linewidth=1.8, 
                     label=f"Cut Threshold $\\epsilon'$ = {eps_prime:.2f}")

    # Set required axis labels
    ax_reach.set_xlabel('Ordering of points (OPTICS Process Order)', fontsize=12, fontweight='bold', labelpad=8)
    ax_reach.set_ylabel('Reachability score / distance', fontsize=12, fontweight='bold', labelpad=8)
    ax_reach.set_title('Reachability Plot (Valleys = Dense Clusters)', fontsize=14, fontweight='bold', pad=12)
    ax_reach.set_xlim(-1, n_points)
    ax_reach.set_ylim(0, cap_val * 1.08)
    ax_reach.grid(axis='y', linestyle=':', alpha=0.6)

    # Annotations for college presentation
    # Find minimum of the reachability plot to place valley pointer
    if finite_reaches:
        min_reach_val = min(finite_reaches)
        min_reach_idx = ordered_reachability.index(min_reach_val)
        ax_reach.annotate(
            'VALLEY:\nDense Cluster\n(Low reachability)',
            xy=(min_reach_idx, min_reach_val),
            xytext=(min_reach_idx, min_reach_val + cap_val * 0.35),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#2b7bba", lw=1.2),
            fontsize=9,
            ha='center'
        )

    # Find highest peak in the middle to point to cluster transition / noise
    peaks = [i for i, r in enumerate(plot_reachabilities) if r >= eps_prime and i > 5 and i < n_points - 5]
    if peaks:
        peak_idx = peaks[0]
        ax_reach.annotate(
            'PEAK:\nCluster Boundary / Noise\n(High reachability)',
            xy=(peak_idx, plot_reachabilities[peak_idx]),
            xytext=(min(peak_idx + 25, n_points - 10), cap_val * 0.95),
            arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f2", ec="#d9534f", lw=1.2),
            fontsize=9,
            ha='center'
        )

    ax_reach.legend(loc='upper right', framealpha=0.9)

    # -------------------------------------------------------------
    # SUBPLOT 2: 2D DATA SPACE CLUSTER PLOT
    # -------------------------------------------------------------
    # Plot noise points first in background
    noise_indices = [i for i in range(len(data)) if labels[i] == -1]
    if noise_indices:
        ax_space.scatter(
            data[noise_indices, 0], data[noise_indices, 1],
            c=[cluster_colors[-1]], marker='x', s=45, label=f"Noise ({len(noise_indices)} pts)",
            linewidths=1.2
        )

    # Plot each cluster
    for c_id in cluster_ids:
        c_pts = [i for i in range(len(data)) if labels[i] == c_id]
        ax_space.scatter(
            data[c_pts, 0], data[c_pts, 1],
            color=[cluster_colors[c_id]], marker='o', s=55, edgecolors='k', linewidths=0.5,
            label=f"Cluster {c_id + 1} ({len(c_pts)} pts)"
        )

    ax_space.set_xlabel('Feature 1 (X coordinate)', fontsize=12, fontweight='bold', labelpad=8)
    ax_space.set_ylabel('Feature 2 (Y coordinate)', fontsize=12, fontweight='bold', labelpad=8)
    ax_space.set_title(f'Extracted Clusters in 2D Space ({n_clusters} Clusters)', fontsize=14, fontweight='bold', pad=12)
    ax_space.grid(True, linestyle=':', alpha=0.6)
    ax_space.legend(loc='best', framealpha=0.9, fontsize=9)

    # Overall Super Title
    suptitle_text = "OPTICS Algorithm Demonstration (Built From Scratch)"
    if dataset_title:
        suptitle_text += f" - {dataset_title}"
    fig.suptitle(suptitle_text, fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save a high-resolution copy for presentation slides
    output_filename = "optics_presentation_result.png"
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    print(f"\n[+] High-resolution presentation plot saved as: '{output_filename}'")
    
    plt.show()


# =====================================================================
# 5. USER INTERFACE & PRESENTATION RUNNER
# =====================================================================

def prompt_user_parameters(suggested):
    """
    Prompts the user for parameters with defaults, allowing seamless
    live presentation demonstration in college.
    """
    print("\n-------------------------------------------------------------")
    print("  PARAMETER INPUT (Press [Enter] to use recommended default)")
    print("-------------------------------------------------------------")

    # Epsilon input
    while True:
        eps_input = input(f"Enter Epsilon (generating radius) [default: {suggested['eps']}]: ").strip()
        if not eps_input:
            eps = float(suggested['eps'])
            break
        try:
            eps = float(eps_input)
            if eps <= 0:
                print("Epsilon must be positive (> 0). Try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # MinPoints input
    while True:
        min_pts_input = input(f"Enter MinPoints (density threshold) [default: {suggested['min_pts']}]: ").strip()
        if not min_pts_input:
            min_pts = int(suggested['min_pts'])
            break
        try:
            min_pts = int(min_pts_input)
            if min_pts < 1:
                print("MinPoints must be at least 1. Try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # Cluster extraction threshold (eps_prime) input
    while True:
        eps_prime_input = input(
            f"Enter Valley Cut Threshold (eps_prime <= eps) [default: {suggested['eps_prime']}]: "
        ).strip()
        if not eps_prime_input:
            eps_prime = float(suggested['eps_prime'])
            break
        try:
            eps_prime = float(eps_prime_input)
            if eps_prime <= 0:
                print("Threshold must be positive (> 0). Try again.")
                continue
            if eps_prime > eps:
                print(f"Warning: eps_prime ({eps_prime}) is greater than eps ({eps}). In OPTICS, eps_prime <= eps.")
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    return eps, min_pts, eps_prime


def main():
    print("======================================================================")
    print("      OPTICS DENSITY-BASED CLUSTERING ALGORITHM (FROM SCRATCH)        ")
    print("       College Presentation Demo: Reachability Valleys & Clusters     ")
    print("======================================================================")
    
    available = datasets.get_available_datasets()
    print("\nAvailable Benchmark Datasets:")
    for key, info in available.items():
        print(f"  [{key}] {info['name']}")
    print("  [6] Load Custom CSV Dataset")

    choice = input("\nSelect dataset to test (1-6) [default: 1]: ").strip()
    if not choice:
        choice = "1"

    if choice in available:
        selected = available[choice]
        data, description, suggested = selected['generator']()
        dataset_name = selected['name']
    elif choice == "6":
        filepath = input("Enter path to your CSV file: ").strip().strip('"').strip("'")
        try:
            data, description, suggested = datasets.load_csv_dataset(filepath)
            dataset_name = f"Custom CSV: {filepath}"
        except Exception as e:
            print(f"Error: {e}")
            print("Falling back to Dataset 1 (Varying Density).")
            data, description, suggested = datasets.generate_varying_density()
            dataset_name = "Varying Density Clusters"
    else:
        print("Unrecognized option. Using default: Dataset 1 (Varying Density).")
        data, description, suggested = datasets.generate_varying_density()
        dataset_name = "Varying Density Clusters"

    print("\n-------------------------------------------------------------")
    print("  DATASET OVERVIEW FOR PRESENTATION")
    print("-------------------------------------------------------------")
    print(description)
    print(f"Total Points: {len(data)}")

    # Prompt user for parameters
    eps, min_pts, eps_prime = prompt_user_parameters(suggested)

    print("\n-------------------------------------------------------------")
    print(f"Running OPTICS algorithm with eps = {eps}, MinPts = {min_pts}...")
    print("100% from-scratch neighborhood search and reachability ordering...")
    print("-------------------------------------------------------------")

    # Run OPTICS
    ordered_list, reachability, core_distances = run_optics(data, eps, min_pts)
    print("[+] OPTICS Ordering Complete.")

    # Extract clusters using valley cut threshold eps_prime
    labels = extract_clusters_dbscan_style(ordered_list, reachability, core_distances, eps_prime)
    n_clusters = len(set(labels) - {-1})
    n_noise = labels.count(-1)

    print(f"[+] Valley Cut Clustering Complete at eps_prime = {eps_prime}:")
    print(f"    - Clusters detected: {n_clusters}")
    print(f"    - Noise points:     {n_noise}")

    # Plot
    print("\nGenerating Reachability Plot and 2D Cluster Space...")
    plot_optics_presentation(data, ordered_list, reachability, labels, eps_prime, dataset_title=dataset_name)


if __name__ == "__main__":
    main()