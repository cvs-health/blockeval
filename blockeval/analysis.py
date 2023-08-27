import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def block_summary(
    campaign_data,
    outcome='outcome',
    block='block',
    treatment='treatment',
    keep_cols=None,
):

    """
    Generates a summary of treatment effects and related statistics for each block in the campaign data.
    
    Args:
        campaign_data (pandas.DataFrame): The DataFrame containing campaign data.
        outcome (str): Name of the column representing the outcome variable.
        block (str): Name of the column representing the block variable.
        treatment (str): Name of the column representing the treatment variable.
        keep_cols (list or str, optional): A list of column names or a single column name
                                           that should remain invariant within blocks. Defaults to None.
                                           If specified, ensures that the values in these columns are
                                           invariant within each block.
    
    Returns:
        pandas.DataFrame: A DataFrame summarizing treatment effects and related statistics for each block.
                          The DataFrame contains the following columns:
                          - 'block': The block identifier.
                          - 'eff': Treatment effect (treated_mean - control_mean).
                          - 'treated_mean': Mean outcome for treated units in the block.
                          - 'control_mean': Mean outcome for control units in the block.
                          - 'treated_size': Number of treated units in the block.
                          - 'control_size': Number of control units in the block.
                          - 'block_size': Total number of units in the block.
                          - 'treatment_proba': Probability of receiving treatment within the block.
    
    Raises:
        ValueError: If values in the specified keep_cols are not invariant within blocks.
    """


    # Check keep_cols is valid
    if keep_cols:
        if type(keep_cols)==str: keep_cols = [keep_cols]
        # Check that the keep_cols do not affect the block
        size_1 = campaign_data.groupby(block, observed=True).size()
        size_2 = campaign_data.groupby([block] + keep_cols, observed=True).size()
        n_level = len(keep_cols)
        level_list = list(range(-n_level, 0))
        size_2 = size_2.reset_index(level=level_list, drop=True)
        are_equal = size_1.equals(size_2)
        if not are_equal:
            raise ValueError(f"Values in the columns {keep_cols} must be invariant within blocks.")
    else:
        keep_cols = []
        
    # Outcome summary for each block
    block_summary = campaign_data.groupby([block] + keep_cols + [treatment], observed=True)[outcome].agg(['size', 'mean']).unstack()
    block_summary.columns = ['control_size', 'treated_size', 'control_mean', 'treated_mean']
    block_summary = (block_summary.eval('eff = treated_mean - control_mean')
                     .eval('block_size = treated_size + control_size'))
    # Treatment information for each block
    treatment_info = campaign_data.groupby([block] + keep_cols, observed=True)[treatment].agg(['mean']).rename(columns={'mean': 'treatment_proba'})
    block_summary = block_summary.merge(treatment_info, left_index=True, right_index=True)
    
    # order columns
    ordered_cols = ['eff', 'treated_mean', 'control_mean', 'treated_size', 'control_size', 'block_size', 'treatment_proba']
    block_summary = block_summary[ordered_cols].reset_index()
        
    return block_summary


def weighted_avg_test(
    campaign_data,
    group_by=None,
    outcome='outcome',
    block='block',
    treatment='treatment',
    use_treated_weights=False,
    alpha = 0.05
):

    """
    Performs a weighted average test to assess treatment effects across different groups or blocks.
    
    Args:
        campaign_data (pandas.DataFrame): The DataFrame containing campaign data.
        group_by (list or str, optional): A list of column names or a single column name by which
                                          to group the analysis. If None, treatment effects will be
                                          assessed for all blocks. Defaults to None.
        outcome (str): Name of the column representing the outcome variable.
        block (str): Name of the column representing the block variable.
        treatment (str): Name of the column representing the treatment variable.
        use_treated_weights (bool, optional): If True, uses treated group size for weights.
                                              If False, uses total block size for weights. Defaults to False.
        alpha (float, optional): The significance level for calculating confidence intervals.
                                Defaults to 0.05.
    
    Returns:
        pandas.DataFrame: A DataFrame summarizing weighted average treatment effects and related statistics
                          for each group or block. The DataFrame contains the following columns:
                          - 'group': The group or block identifier(s).
                          - 'eff': Weighted average treatment effect (ATE) across the group or block.
                          - 'treated_mean': Weighted average mean outcome for treated units in the group or block.
                          - 'control_mean': Weighted average mean outcome for control units in the group or block.
                          - 'treated_size': Total number of treated units in the group or block.
                          - 'control_size': Total number of control units in the group or block.
                          - 'group_size': Total number of units in the group or block.
                          - 'eff_se': Standard error of the weighted ATE.
                          - 'z': z-score of the ATE.
                          - 'p_value': Two-sided p-value for the ATE.
                          - 'ci_low': Lower bound of the confidence interval for the ATE.
                          - 'ci_upp': Upper bound of the confidence interval for the ATE.
                          - 'incremental': Incremental effect, calculated based on treated weights
                                          or total block size weights.
    """

    # prepare data
    if group_by:
        if type(group_by)==str: group_by = [group_by]
        short_group_by = [v for v in group_by if v!='block'] # remove block from group_by if exist
        summary_cols = [block] + short_group_by + [treatment]
    else:
        summary_cols = [block, treatment] # the user wants the treatment effect for all the blocks
        group_by = ['group'] # column used in the summary to group all the blocks in a single group
        
    # outcome summary for each block
    block_summary = campaign_data.groupby(summary_cols, observed=True)[outcome].agg(['size', 'mean', 'var']).unstack() # group by blocks
    col_names = [str(pair[1]) + '_' + pair[0] for pair in block_summary.columns.values]
    block_summary.columns = [name.replace('0_', 'control_').replace('1_', 'treated_') for name in col_names] # control_size, treated_size, control_mean, treated_mean, ...
    block_summary = (block_summary.eval('eff = treated_mean - control_mean')
                     .eval('eff_var = treated_var/treated_size + control_var/control_size')
                     .eval('eff_se = sqrt(eff_var)')
                     .assign(group='all') # create column to group all the blocks in a single group
                     .reset_index())
    
    if use_treated_weights:
        block_summary['selected_size'] = block_summary['treated_size'] # uses treated group size for weights
    else:
        block_summary['selected_size'] = block_summary['treated_size'] + block_summary['control_size'] # uses block size for weights
            
    groups = block_summary.groupby(group_by, observed=True)  
    results = []
    all_keys = []
    for group_key, group_df in groups:
        
        if isinstance(group_key, str): group_key = (group_key,) # make sure it is always a tuple

        group_df = group_df.assign(norm_weight=group_df['selected_size'] / group_df['selected_size'].sum())

        # Calculate the weighted average treatment effect (ATE) by multiplying normalized block sizes by treatment effects and summing them up.
        weighted_ate = (group_df['norm_weight'] * group_df['eff']).sum()

        # Calculate the weighted average treatment and control mean
        weighted_control_mean = (group_df['norm_weight'] * group_df['control_mean']).sum()
        weighted_treated_mean = (group_df['norm_weight'] * group_df['treated_mean']).sum()

        # Calculate the weighted ATE variance by multiplying squared normalized block sizes by treatment effect variances and summing them up.
        weighted_ate_var = (group_df['norm_weight'] ** 2 * group_df['eff_var']).sum()
        # Calculate the standard deviation of the weighted ATE.
        weighted_ate_sd = np.sqrt(weighted_ate_var)

        # Get total treatment and control group size
        total_treated_size = group_df['treated_size'].sum()
        total_ctrl_size = group_df['control_size'].sum()
        total_block_size = total_treated_size + total_ctrl_size

        # Compute the confidence interval (CI) bounds and two-sided p-value.
        z_score = stats.norm.ppf(1 - alpha / 2)  # z-score for 95% confidence level
        lower_bound = weighted_ate - z_score * weighted_ate_sd
        upper_bound = weighted_ate + z_score * weighted_ate_sd
        z_test = weighted_ate / weighted_ate_sd  # z-score
        p_value = 2 * (1 - stats.norm.cdf(abs(z_test)))  # two-sided p-value

        row_dict = {
            'eff': weighted_ate,
            'treated_mean': weighted_treated_mean,
            'control_mean': weighted_control_mean,
            'treated_size': total_treated_size,
            'control_size': total_ctrl_size,
            'group_size': total_block_size,
            'eff_se': weighted_ate_sd,
            'z': z_test,
            'p_value': p_value,
            'ci_low': lower_bound,
            'ci_upp': upper_bound,
        }
        
        all_keys.append(group_key)
        results.append(row_dict)
        
    labels = pd.DataFrame(all_keys, columns=group_by)
    results = pd.concat([labels, pd.DataFrame(results)], axis=1)
    
    if use_treated_weights:
        results['incremental'] = results['eff']*results['treated_size']
    else:
        results['incremental'] = results['eff']*results['group_size']

    return results


def comparison_test(
    campaign_data,
    compare_along,
    outcome='outcome',
    block='block',
    treatment='treatment',
    use_treated_weights=False,
    alpha = 0.05
):
    
    """
    Performs a comparison test to assess treatment effects between different groups along a specified dimension.
    
    Args:
        campaign_data (pandas.DataFrame): The DataFrame containing campaign data.
        compare_along (str): The column along which treatment effects are to be compared (e.g., 'group').
        outcome (str): Name of the column representing the outcome variable.
        block (str): Name of the column representing the block variable.
        treatment (str): Name of the column representing the treatment variable.
        use_treated_weights (bool, optional): If True, uses treated group size for weights.
                                              If False, uses total block size for weights. Defaults to False.
        alpha (float, optional): The significance level for calculating confidence intervals.
                                Defaults to 0.05.
    
    Returns:
        pandas.DataFrame: A DataFrame summarizing the comparison test results for treatment effects between
                          different groups along the specified dimension. The DataFrame contains the following columns:
                          - 'eff_delta': Difference in weighted average treatment effects between variant and reference groups.
                          - 'variant_grp': The variant group identifier.
                          - 'reference_grp': The reference group identifier.
                          - 'variant_size': Total number of units in the variant group.
                          - 'reference_size': Total number of units in the reference group.
                          - 'eff_se': Standard error of the difference in treatment effects.
                          - 'z': z-score of the difference.
                          - 'p_value': Two-sided p-value for the difference.
                          - 'ci_low': Lower bound of the confidence interval for the difference.
                          - 'ci_upp': Upper bound of the confidence interval for the difference.
    """

    # prepare data
    if compare_along==block:
        summary_cols = [block, treatment]
    else:
        summary_cols = [block, compare_along, treatment]

    # outcome summary for each block
    block_summary = campaign_data.groupby(summary_cols, observed=True)[outcome].agg(['size', 'mean', 'var']).unstack() # group by blocks
    col_names = [str(pair[1]) + '_' + pair[0] for pair in block_summary.columns.values]
    block_summary.columns = [name.replace('0_', 'control_').replace('1_', 'treated_') for name in col_names] # control_size, treated_size, control_mean, treated_mean, ...
    block_summary = (block_summary.eval('eff = treated_mean - control_mean')
                     .eval('eff_var = treated_var/treated_size + control_var/control_size')
                     .eval('eff_se = sqrt(eff_var)')
                     .assign(group='all') # create column to group all the blocks in a single group
                     .reset_index())
    
    if use_treated_weights:
        block_summary['selected_size'] = block_summary['treated_size'] # uses treated group size for weights
    else:
        block_summary['selected_size'] = block_summary['treated_size'] + block_summary['control_size'] # uses block size for weights

        
    groups = block_summary.groupby(compare_along, observed=True)
    group_results = []
    all_keys = []   
    for group_key, group_df in groups:

        # Calculate the weighted average treatment effect (ATE) by multiplying normalized block sizes by treatment effects and summing them up.
        group_df = group_df.assign(norm_weight=group_df['selected_size'] / group_df['selected_size'].sum())
        weighted_ate = (group_df['norm_weight'] * group_df['eff']).sum()

        # Calculate the weighted average treatment and control mean
        weighted_control_mean = (group_df['norm_weight'] * group_df['control_mean']).sum()
        weighted_treated_mean = (group_df['norm_weight'] * group_df['treated_mean']).sum()

        # Calculate the weighted ATE variance by multiplying squared normalized block sizes by treatment effect variances and summing them up.
        weighted_ate_var = (group_df['norm_weight'] ** 2 * group_df['eff_var']).sum()

        # Get total treatment and control group size (optional)
        total_treated_size = group_df['treated_size'].sum()
        total_ctrl_size = group_df['control_size'].sum()
        total_block_size = total_treated_size + total_ctrl_size

        row_dict = {
            'eff': weighted_ate,
            'treated_mean': weighted_treated_mean,
            'control_mean': weighted_control_mean,
            'treated_size': total_treated_size,
            'control_size': total_ctrl_size,
            'group_size': total_block_size,
            'eff_var': weighted_ate_var,            
        }
        
        all_keys.append(group_key)
        group_results.append(row_dict)
    
    group_results = pd.DataFrame(group_results, index=all_keys) # row index is the group label
    
    # calculate ATE delta between groups and reference group
    reference_grp = campaign_data[compare_along].cat.categories[0]
    reference_result = group_results.loc[reference_grp, :]
    reference_grp_size = reference_result['group_size']
    
    delta_results = []
    for variant_grp, variant_result in group_results.iloc[1:].iterrows():
        eff_delta = variant_result['eff'] - reference_result['eff']
        variant_grp_size = variant_result['group_size']
        eff_var = variant_result['eff_var'] + reference_result['eff_var']
        eff_se = np.sqrt(eff_var)
        
        # Compute the confidence interval (CI) bounds and two-sided p-value.
        alpha = 0.05
        z_score = stats.norm.ppf(1 - alpha / 2)  # z-score for 95% confidence level
        lower_bound = eff_delta - z_score * eff_se
        upper_bound = eff_delta + z_score * eff_se
        z_test = eff_delta / eff_se  # z-score
        p_value = 2 * (1 - stats.norm.cdf(abs(z_test)))  # two-sided p-value

        row_dict = {
            'eff_delta': eff_delta,
            'variant_grp': variant_grp,
            'reference_grp': reference_grp,
            'variant_size': int(variant_grp_size),
            'reference_size': int(reference_grp_size),
            'eff_se': eff_se,
            'z': z_test,
            'p_value': p_value,
            'ci_low': lower_bound,
            'ci_upp': upper_bound,            
        }

        delta_results.append(row_dict) # list of dicts
    
    delta_results = pd.DataFrame(delta_results)
    ordered_cols = ['eff_delta', 'variant_grp', 'reference_grp', 'variant_size', 'reference_size', 'eff_se', 'z', 'p_value', 'ci_low', 'ci_upp']
    delta_results = delta_results[ordered_cols]
    
    return delta_results


def weighted_avg_bootstrap(
    campaign_data,
    group_by=None,
    outcome='outcome',
    block='block',
    treatment='treatment',
    use_treated_weights=False,
    alpha=0.05,
    n_bootstrap=2000
):

    """
    Performs a weighted average bootstrap test to assess treatment effects across different groups or blocks.

    Args:
        campaign_data (pandas.DataFrame): The DataFrame containing campaign data.
        group_by (list or str, optional): A list of column names or a single column name by which
                                          to group the analysis. If None, treatment effects will be
                                          assessed for all blocks. Defaults to None.
        outcome (str): Name of the column representing the outcome variable.
        block (str): Name of the column representing the block variable.
        treatment (str): Name of the column representing the treatment variable.
        use_treated_weights (bool, optional): If True, uses treated group size for weights.
                                              If False, uses total block size for weights. Defaults to False.
        alpha (float, optional): The significance level for calculating confidence intervals.
                                Defaults to 0.05.
        n_bootstrap (int, optional): The number of bootstrap iterations. Defaults to 2000.

    Returns:
        pandas.DataFrame: A DataFrame summarizing bootstrap estimates and p-values for treatment effects between
                          different groups. The DataFrame contains the following columns:
                          - 'eff': Bootstrap mean of the treatment effect.
                          - 'ci_low': Lower bound of the bootstrap confidence interval for the effect.
                          - 'ci_upp': Upper bound of the bootstrap confidence interval for the effect.
                          - 'p_value': Bootstrap p-value for the effect.
    """
        
    def permute_treatment_within_blocks(df, block_name, treatment):
        # shuffle df in place
        for b in block_name:
            mask = df[block] == b
            treatment_values = df.loc[mask, treatment].values # convert to array
            np.random.shuffle(treatment_values) # shuffle in place
            df.loc[mask, treatment] = treatment_values
        
    # prepare data
    if group_by:
        if type(group_by)==str: group_by = [group_by]
        short_group_by = [v for v in group_by if v!='block'] # remove block from group_by
        summary_cols = [block] + short_group_by + [treatment]
    else:
        summary_cols = [block, treatment]
        group_by = ['group'] # column used in the summary to group all the blocks in a single group
            
    # block sizes (remain constant accross bootstrap)
    block_summary = campaign_data.groupby(summary_cols, observed=True)[outcome].agg(['size', 'mean']).unstack()
    block_summary.columns = ['control_size', 'treated_size', 'control_mean', 'treated_mean']
    block_summary = (block_summary.eval('eff = treated_mean - control_mean')
                    .assign(group='all') # create column to group all the blocks in a single group
                    .reset_index())
    if use_treated_weights:
        block_summary['selected_size'] = block_summary['treated_size'] # uses treated group size for weights
    else:
        block_summary['selected_size'] = block_summary['treated_size'] + block_summary['control_size'] # uses block size for weights
        
    # observed ATE
    grouped_summary = block_summary.groupby(group_by, observed=True)
    observed_ate = grouped_summary.apply(lambda x: np.average(x['eff'], weights=x['selected_size'])) # return a series with groups as index

    # loop through bootstrap
    bootstrap_ates = []
    grouped_campaign = campaign_data.groupby(summary_cols, observed=True)
    for i in range(n_bootstrap):
        # bootstrap design
        bootstrapped_data = pd.concat([group.sample(n=len(group), replace=True) for _, group in grouped_campaign], axis=0).reset_index() # sample within block/treatment
        # summarize blocks
        bootstrapped_summary = bootstrapped_data.groupby(summary_cols, observed=True)[outcome].agg(['mean']).unstack()
        bootstrapped_summary.columns = ['control_mean', 'treated_mean']
        bootstrapped_summary = bootstrapped_summary.reset_index(drop=True)
        block_summary['eff'] = bootstrapped_summary['treated_mean'] - bootstrapped_summary['control_mean'] # modify existing block_summary

        # boostrapped ATE
        grouped_summary = block_summary.groupby(group_by, observed=True)
        ate_by_group = grouped_summary.apply(lambda x: np.average(x['eff'], weights=x['selected_size'])) # return a series
        # add bootstrap
        bootstrap_ates.append(ate_by_group) # treatment effect for each group of blocks
            
    bootstrap_estim = (
        pd.DataFrame(bootstrap_ates) # groups are in columns
        .agg(['mean', lambda x: x.quantile(alpha/2), lambda x: x.quantile(1-alpha/2)]) # metrics are in rows
        .reset_index(drop=True)
        .rename(index={0: 'eff', 1: 'ci_low', 2: 'ci_upp'}) # groups are in row, metrics are in columns
        .transpose()
    )
    
    # loop through permutation
    permutation_ates = []
    block_name = campaign_data[block].unique()
    permuted_data = campaign_data.copy()
    for i in range(n_bootstrap):
        permute_treatment_within_blocks(permuted_data, block_name, treatment) # shuffle permuted_data in place within blocks
        # summarize blocks
        permuted_summary = permuted_data.groupby(summary_cols, observed=True)[outcome].agg(['mean']).unstack()
        permuted_summary.columns = ['control_mean', 'treated_mean']
        permuted_summary = permuted_summary.reset_index(drop=True)
        block_summary['eff'] = permuted_summary['treated_mean'] - permuted_summary['control_mean']
         # group blocks
        grouped_summary = block_summary.groupby(group_by, observed=True)
        ate_by_group = grouped_summary.apply(lambda x: np.average(x['eff'], weights=x['selected_size'])) # return a series
        # add permutation
        permutation_ates.append(ate_by_group) # treatment effect for each group of blocks
        
    permutation_ates = pd.DataFrame(permutation_ates) # groups are in columns
    extreme_count = np.sum(permutation_ates >= observed_ate)
    p_value = 2 * extreme_count / len(permutation_ates)
    p_value.name = 'p_value'
    bootstrap_estim = pd.concat([bootstrap_estim, p_value], axis=1)

    return bootstrap_estim
    

def comparison_bootstrap(
    campaign_data,
    compare_along,
    outcome='outcome',
    block='block',
    treatment='treatment',
    use_treated_weights=False,
    alpha=0.05,
    n_bootstrap=2000
):

    """
    Performs a comparison bootstrap test to assess treatment effects between different groups along a specified dimension.

    Args:
        campaign_data (pandas.DataFrame): The DataFrame containing campaign data.
        compare_along (str): The column along which treatment effects are to be compared (e.g., 'group').
        outcome (str): Name of the column representing the outcome variable.
        block (str): Name of the column representing the block variable.
        treatment (str): Name of the column representing the treatment variable.
        use_treated_weights (bool, optional): If True, uses treated group size for weights.
                                              If False, uses total block size for weights. Defaults to False.
        alpha (float, optional): The significance level for calculating confidence intervals.
                                Defaults to 0.05.
        n_bootstrap (int, optional): The number of bootstrap iterations. Defaults to 2000.

    Returns:
        pandas.DataFrame: A DataFrame summarizing bootstrap estimates, p-values, and confidence intervals
                          for treatment effect comparisons between different groups. The DataFrame contains the following columns:
                          - 'eff_delta': Bootstrap mean of the difference in treatment effects.
                          - 'variant_grp': The variant group identifier.
                          - 'reference_grp': The reference group identifier.
                          - 'p_value': Bootstrap p-value for the difference.
                          - 'ci_low': Lower bound of the bootstrap confidence interval for the difference.
                          - 'ci_upp': Upper bound of the bootstrap confidence interval for the difference.
    """
    
    def permute_treatment_within_blocks(df, block_name, treatment):
        # shuffle df in place
        for b in block_name:
            mask = df[block] == b
            treatment_values = df.loc[mask, treatment].values # convert to array
            np.random.shuffle(treatment_values) # shuffle in place
            df.loc[mask, treatment] = treatment_values
        
    # prepare data
    if compare_along==block:
        summary_cols = [block, treatment]
    else:
        summary_cols = [block, compare_along, treatment]
        
    # block size summary (remain constant accross bootstrap)
    block_summary = campaign_data.groupby(summary_cols, observed=True)[outcome].agg(['size', 'mean']).unstack() # group by blocks
    block_summary.columns = ['control_size', 'treated_size', 'control_mean', 'treated_mean']
    block_summary = (block_summary.eval('eff = treated_mean - control_mean') # treatment effect by blocks
                    .reset_index())
    if use_treated_weights:
        block_summary['selected_size'] = block_summary['treated_size'] # uses treated group size for weights
    else:
        block_summary['selected_size'] = block_summary['treated_size'] + block_summary['control_size'] # uses block size for weights
                
    # observed treatment effect
    grouped_summary = block_summary.groupby(compare_along, observed=True) # group by groups
    group_results = grouped_summary.apply(lambda x: np.average(x['eff'], weights=x['selected_size'])) # treatment effect by groups (series)
    
    # observed deltas
    reference_grp = campaign_data[compare_along].cat.categories[0] # TODO, check that the first is always the reference
    reference_result = group_results[reference_grp]
    observed_deltas = pd.Series()
    
    # calculate observed ATE deltas between groups and reference group
    for variant_grp, variant_result in group_results[1:].iteritems(): # remove first (TODO, check that the first is always the reference)
        eff_delta = variant_result - reference_result       
        observed_deltas = observed_deltas.append(pd.Series({variant_grp: eff_delta}))
                
    # loop through bootstrap
    bootstrap_deltas = []
    grouped_campaign = campaign_data.groupby(summary_cols, observed=True) # group by blocks
    for i in range(n_bootstrap):
        # bootstrap design
        bootstrapped_data = pd.concat([group.sample(n=len(group), replace=True) for _, group in grouped_campaign], axis=0).reset_index()
        # summarize blocks
        bootstrapped_summary = bootstrapped_data.groupby(summary_cols, observed=True)[outcome].agg(['mean']).unstack() # group by blocks
        bootstrapped_summary.columns = ['control_mean', 'treated_mean']
        bootstrapped_summary = bootstrapped_summary.reset_index(drop=True)
        block_summary['eff'] = bootstrapped_summary['treated_mean'] - bootstrapped_summary['control_mean'] # treatment effect by blocks

        # boostrapped ATE
        grouped_summary = block_summary.groupby(compare_along, observed=True) # group by groups
        group_results = grouped_summary.apply(lambda x: np.average(x['eff'], weights=x['selected_size'])) # treatment effect by groups
        
        # deltas
        reference_result = group_results[reference_grp]
        deltas = pd.Series()
        # calculate ATE deltas between groups and reference group
        for variant_grp, variant_result in group_results[1:].iteritems(): # remove first (TODO, check that the first is always the reference)
            eff_delta = variant_result - reference_result       
            deltas = deltas.append(pd.Series({variant_grp: eff_delta}))
        
        # add bootstrap
        bootstrap_deltas.append(deltas) # deltas is a series        
    
    # collect bootstrap results
    bootstrap_estim = (
        pd.DataFrame(bootstrap_deltas) # variant names in columns
        .agg(['mean', lambda x: x.quantile(alpha/2), lambda x: x.quantile(1-alpha/2)]) # metrics are in rows
        .reset_index(drop=True)
        .rename(index={0: 'eff_delta', 1: 'ci_low', 2: 'ci_upp'}) # groups are in row, metrics are in columns
        .transpose()
    )
        
    # loop through permutation
    permutation_deltas = []
    block_name = campaign_data[block].unique()
    permuted_data = campaign_data.copy()
    for i in range(n_bootstrap):
        # permute design
        permute_treatment_within_blocks(permuted_data, block_name, treatment) # shuffle in place
        # summarize blocks
        permuted_block_summary = permuted_data.groupby(summary_cols, observed=True)[outcome].agg(['mean']).unstack() # group by blocks
        permuted_block_summary.columns = ['control_mean', 'treated_mean']
        permuted_block_summary = permuted_block_summary.reset_index(drop=True)
        block_summary['eff'] = permuted_block_summary['treated_mean'] - permuted_block_summary['control_mean'] # modify existing block_summary, treatment effect by blocks
         # group blocks
        grouped_summary = block_summary.groupby(compare_along, observed=True) # group by groups
        group_results = grouped_summary.apply(lambda x: np.average(x['eff'], weights=x['selected_size'])) # treatment effect by groups
        
        # deltas
        reference_result = group_results[reference_grp]
        deltas = pd.Series()
        # loop through comparisons
        for variant_grp, variant_result in group_results[1:].iteritems(): # remove first (TODO, check that the first is always the reference)
            eff_delta = variant_result - reference_result       
            deltas = deltas.append(pd.Series({variant_grp: eff_delta}))
        
        # add bootstrap
        permutation_deltas.append(deltas) # deltas is a series        
            
    # collect premutation results
    permutation_deltas = pd.DataFrame(permutation_deltas) # variant names are in columns
    
    extreme_count = np.sum(permutation_deltas >= observed_deltas)
    p_value = 2 * extreme_count / len(permutation_deltas)
    p_value.name = 'p_value'
    bootstrap_estim = pd.concat([bootstrap_estim, p_value], axis=1)
    
    # add variant and reference names
    bootstrap_estim = bootstrap_estim.reset_index().rename(columns={'index': 'variant_grp'})
    bootstrap_estim['reference_grp'] = reference_grp
    bootstrap_estim = bootstrap_estim[['eff_delta', 'variant_grp', 'reference_grp', 'p_value', 'ci_low', 'ci_upp']] # reorder columns    

    return bootstrap_estim


