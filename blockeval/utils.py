import pandas as pd
import numpy as np


def campaign_simulation(
    blocks, 
    block_sizes,
    treatment_probas,
    control_means,
    treatment_effects=None,
    treated_means=None,                    
    control_sds=None,
    treated_sds=None,                            
    random_seed=42
):

    """
    Simulates campaign data for A/B testing based on provided parameters.

    Args:
        blocks (list): List of block identifiers.
        block_sizes (list): List of block sizes corresponding to each block.
        treatment_probas (list): List of treatment probabilities for each block.
        control_means (list): List of means for the control group for each block.
        treatment_effects (list or None, optional): List of treatment effects for each block.
                                                     If None, treated_means will be used to calculate treatment effects.
                                                     Defaults to None.
        treated_means (list or None, optional): List of means for the treated group for each block.
                                                 If None, treatment_effects will be used to calculate treated_means.
                                                 Defaults to None.
        control_sds (list or None, optional): List of standard deviations for the control group for each block.
                                              If provided, Normal distribution is used for outcome generation.
                                              If None, binomial distribution is used for outcome generation.
                                              Defaults to None.
        treated_sds (list or None, optional): List of standard deviations for the treated group for each block.
                                              If provided, Normal distribution is used for outcome generation.
                                              If None, binomial distribution is used for outcome generation.
                                              Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        pandas.DataFrame: A DataFrame containing the simulated campaign data with the following columns:
                          - 'outcome': Outcome variable (0 or 1).
                          - 'treatment': Treatment group identifier (0 for control, 1 for treated).
                          - 'block': Block identifier.
    """

        
    if treatment_effects is None and treated_means is None:
        raise ValueError("Provide treatment_effects or treated_means")
    elif treatment_effects is None:
        treatment_effects = pd.Series(treated_means) - pd.Series(control_means)
    
    np.random.seed(random_seed)
    campaign_data = []
    for i, (block, block_size, treatment_proba, control_mean, treatment_effect) in enumerate(zip(blocks, block_sizes, treatment_probas, control_means, treatment_effects)):
        
        # Create block data
        n_treated = int(treatment_proba*block_size)
        n_control = block_size - n_treated
        treated_mean = control_mean + treatment_effect
        control_mean = control_mean
        if control_sds is None or treated_sds is None:
            # Bernoulli outcome
            treated_outcome = np.random.binomial(1, treated_mean, n_treated)
            control_outcome = np.random.binomial(1, control_mean, n_control)
        else:
            # Normal outcome
            treated_sd = treated_sds[i]
            control_sd = control_sds[i]
            treated_outcome = np.random.normal(treated_mean, treated_sd, n_treated)
            control_outcome = np.random.normal(control_mean, control_sd, n_control)
            
        
        # Concatenate the two arrays into a single array
        treated_data = pd.DataFrame({'outcome': treated_outcome, 'treatment': 1})
        control_data = pd.DataFrame({'outcome': control_outcome, 'treatment': 0})
        block_data = pd.concat([treated_data, control_data])
        block_data["block"] = block
        
        # Append block data to list
        campaign_data.append(block_data)
    
    # Combine all block data into a single dataframe
    campaign_data = pd.concat(campaign_data)
    campaign_data['block'] = pd.Categorical(campaign_data['block'], categories=blocks, ordered=True)

    
    return campaign_data

