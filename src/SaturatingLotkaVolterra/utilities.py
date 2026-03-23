import numpy as np
import pandas as pd
import torch

def generate_presence_absence_initial_conditions(
    samples,
    num_species,
    target_total_OD=0.02,
    device="cpu",
    dtype=torch.float32,
    uniform_community_size=True,
    min_community_size_N=1,
):
    """
    Generate initial conditions with presence/absence patterns for species.

    Args:
        samples (int): Number of samples to generate
        num_species (int): Number of species (N)
        target_total_OD (float): Target total OD for species (default: 0.02)
        device (str): PyTorch device (default: "cpu")
        dtype (torch.dtype): PyTorch data type (default: torch.float32)
        uniform_community_size (bool): If True, sample community sizes uniformly from [min_size, max_size].
                                       If False, use 50% probability for each species (default: True)
        min_community_size_N (int): Minimum number of species present (default: 1)

    Returns:
        torch.Tensor: N0 of shape (samples, num_species) with presence/absence patterns
            and total OD normalized to target_total_OD.
    """
    N0 = torch.zeros((samples, num_species), device=device, dtype=dtype)

    for i in range(samples):
        if uniform_community_size:
            community_size_N = torch.randint(
                min_community_size_N, num_species + 1, (1,), device=device
            ).item()
            indices = torch.randperm(num_species, device=device)[:community_size_N]
            presence_mask_N = torch.zeros(num_species, dtype=torch.bool, device=device)
            presence_mask_N[indices] = True
        else:
            presence_mask_N = torch.rand(num_species, device=device) > 0.5

        num_present_N = presence_mask_N.sum().item()
        if num_present_N > 0:
            N0[i, presence_mask_N] = target_total_OD / num_present_N
        else:
            idx = torch.randint(0, num_species, (1,), device=device)
            N0[i, idx] = target_total_OD

    return N0

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