import numpy as np
import pandas as pd
import torch

def generate_initial_conditions(
    samples,
    num_species,
    target_total_OD=0.02,
    alpha=1.0,
    min_community_size=None,
    max_community_size=None,
    device="cpu",
    dtype=torch.float32,
):
    """
    Unified initial-condition sampler.

    Each sample is constructed in two stages:

      1. Community size k is drawn uniformly from
         [min_community_size, max_community_size]; a random subset of k species
         is marked present. Defaults select k = num_species (full community)
         for every sample.
      2. Per-species ratios among the k present species are drawn from
         Dirichlet(alpha * ones(k)) and scaled so the row sums to
         target_total_OD. Absent species stay at zero. Passing alpha=None
         (or float('inf')) uses an equal split (ratios = 1/k).

    Recipes for the common cases:
      - Full community, Dirichlet ratios (default):
            alpha=1.0 (uniform on simplex)
      - Full community, equal OD per species (sanity baseline):
            alpha=None
      - Presence/absence subcommunities with equal OD inside each community:
            alpha=None,
            min_community_size=1, max_community_size=num_species

    Args:
        samples (int): Number of samples to generate.
        num_species (int): Number of species (N).
        target_total_OD (float): Per-sample total OD (default: 0.02).
        alpha (float | None): Dirichlet concentration (default: 1.0, uniform
            on the simplex). None / float('inf') -> equal split across the
            present species.
        min_community_size (int | None): Smallest community size, inclusive.
            Defaults to num_species (always full community).
        max_community_size (int | None): Largest community size, inclusive.
            Defaults to num_species.
        device (str): PyTorch device (default: "cpu").
        dtype (torch.dtype): PyTorch data type (default: torch.float32).

    Returns:
        torch.Tensor of shape (samples, num_species). Rows sum to
        target_total_OD, with zeros for absent species.
    """
    if min_community_size is None:
        min_community_size = num_species
    if max_community_size is None:
        max_community_size = num_species
    if not 1 <= min_community_size <= max_community_size <= num_species:
        raise ValueError(
            f"require 1 <= min_community_size={min_community_size} <= "
            f"max_community_size={max_community_size} <= num_species={num_species}"
        )

    # --- Presence mask (samples, num_species): k ones per row, k drawn per-sample.
    if min_community_size == num_species:
        present = torch.ones((samples, num_species), device=device, dtype=dtype)
    else:
        k_vec = torch.randint(
            min_community_size, max_community_size + 1, (samples,), device=device
        )
        rand_scores = torch.rand((samples, num_species), device=device)
        sorted_scores, _ = rand_scores.sort(dim=-1)
        # k-th smallest score per row -> threshold for selecting k present species
        thresholds = sorted_scores.gather(1, (k_vec - 1).unsqueeze(-1))
        present = (rand_scores <= thresholds).to(dtype)

    # --- Ratios across present species
    equal_split = alpha is None or alpha == float("inf")
    if equal_split:
        ratios = present / present.sum(dim=-1, keepdim=True)
    else:
        alpha_vec = torch.full((num_species,), float(alpha), device=device, dtype=dtype)
        full = torch.distributions.Dirichlet(alpha_vec).sample((samples,)).to(dtype=dtype)
        masked = full * present
        ratios = masked / masked.sum(dim=-1, keepdim=True)

    return ratios * target_total_OD

def create_dataframe(species_names, t_eval, solution, treatment_ids, replicate_ids=None, exp_ids=None):
    """
    Creates a DataFrame from the given species and mediator names, time points, solution, treatment IDs, replicate IDs, and experiment IDs.
    
    Args:
        species_names (list): List of species names.
        t_eval (array): Array of time points.
        solution (array): Array of solution data.
        treatment_ids (array): Array of treatment IDs.
        replicate_ids (array, optional): Array of replicate IDs. Defaults to None.
        exp_ids (array, optional): Array of experiment IDs. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame containing the species and mediator data, time points, treatment IDs, replicate IDs, and experiment IDs.
    """
    # Create treatment_names
    N_species = len(species_names)
    treatment_ids = np.tile(treatment_ids, (solution.shape[1], 1)).T.flatten()
    treatment_ids_df = pd.DataFrame(treatment_ids, columns=["Treatments"])

    # Create dataframe data
    if replicate_ids is not None:
        replicate_ids = np.tile(replicate_ids, (solution.shape[1], 1)).T.flatten()[:len(treatment_ids)]
        replicate_ids_df = pd.DataFrame(replicate_ids, columns=["Replicates"])
    if exp_ids is not None:
        exp_ids = np.tile(exp_ids, (solution.shape[1], 1)).T.flatten()
        exp_ids_df = pd.DataFrame(exp_ids, columns=["Experiments"])
    time_points = np.tile(t_eval, (solution.shape[0], 1)).flatten()
    sp_mat = solution.reshape((solution.shape[0] * solution.shape[1], N_species))

    # Create dataframe
    sp_df = pd.DataFrame(sp_mat, columns=species_names)
    time_df = pd.DataFrame(time_points, columns=["Time"])
    df = pd.concat([time_df, treatment_ids_df, sp_df], axis=1)
    if replicate_ids is not None:
        df = pd.concat([df, replicate_ids_df], axis=1)
    if exp_ids is not None:
        df = pd.concat([df, exp_ids_df], axis=1)
    return df