#!/apps/anaconda3/bin/python
"""
CAP6 Ensemble Delayed Action Cluster Script - SGE Array Job Version

This script combines Latin Hypercube Sampling (LHS) parameter exploration with
delayed action analysis on a cluster environment. It runs BOTH optimal and 
delayed action scenarios for each parameter sample, comparing them using the 
ConstraintAnalysis class.

The workflow:
1. Generate/load LHS parameter samples (RA, EIS, PRTP, tech_chg, tech_scale)
2. SGE_TASK_ID maps to unique (sample_index, delay_year) combinations
3. Each task runs optimal and delayed scenarios for one (sample, delay) pair
4. Use ConstraintAnalysis to calculate deadweight costs

Mapping:
    For N_SAMPLES samples and delay_years=[5,10,15], you get N_SAMPLES*3 combinations:
    Task 1  -> (sample=0, delay_year=5)
    Task 2  -> (sample=0, delay_year=10)
    Task 3  -> (sample=0, delay_year=15)
    Task 4  -> (sample=1, delay_year=5)
    Task 5  -> (sample=1, delay_year=10)
    ...

Usage:
    # Configure parameters below, then submit array job.
    # Samples will be generated automatically if they don't exist.
    # For N_SAMPLES=100 and delay_years=[5,10,15] -> 300 tasks:
    grid_run --grid_mem=100G --grid_submit=batch --grid_array=1-300 \\
             --grid_ncpus=4 ./run_ensemble_delayed_array_job.sh

Environment Variables Expected:
    SGE_TASK_ID: Integer from 1 to (N_SAMPLES * len(delay_years))
    OUTPUT_FOLDER: Name of output folder in data/ - optional override
    BASELINE_NUM: SSP baseline scenario (1-5) - optional override

Configuration:
    - Edit `N_SAMPLES` below to set number of LHS samples
    - Edit `delay_years` list below to set which delay years to test (e.g., [5, 10, 15])
    - Edit parameter ranges (ubs, lbs) to customize sampling
    - Edit `baseline_num` for SSP scenario selection

Author: Theo Moers
"""

import os
import sys
import pprint
import numpy as np
import csv
import fcntl
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tree import TreeModel
from src.emit_baseline import BPWEmissionBaseline
from src.cost import BPWCost
from src.climate import BPWClimate
from src.damage import BPWDamage
from src.utility import EZUtility
from src.analysis.climate_output import ClimateOutput
from src.analysis.delayed_action import get_delay_nodes, ConstraintAnalysis
from src.optimization import GeneticAlgorithm, GradientSearch
from src.gen_samples import generate_samples


N_SAMPLES = 3000

delay_years = [5, 10, 15]

DIMS = 7
ubs = [15., 1.08, 3., 3., 0.0147, 20000., 0.025] # Upper bounds
lbs = [3., 0.55, 0., 0., 0.001, 5000., 0.010] # Lower bounds
param_names = ["RA", "EIS", "tech_chg", "tech_scale", "PRTP", "bs_premium", "growth"] 
# Risk Aversion, elasticity of intertemporal substitution, rate of exogeneous technological development, rate of endogeneous technological development, pure rate of time preference, backstop premium, consumption growth rate

# Fixed parameters (not sampled)
baseline_num = 2 # SSP2 baseline
dam_func = 0 # mixed damage function
tip_on = 1 # turn tipping points on
d_unc = 1 # damage uncertainty on
t_unc = 1 # temperature uncertainty on
no_free_lunch = False # backstop no free lunch

output_folder = "ensemble-cb-ir-l-bs-g"

test_mode = False
import_damages = False # set to True if possible to avoid file write conflicts

PROJECT_ROOT = "/user/tlm2160/EZDelay"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

COMMON_YEARS = [2020, 2025, 2030, 2035, 2060, 2110, 2160, 2210, 2250]
START_YEAR = 2020


def calculate_period_climate_metrics(m, tree, damage, climate, emit_baseline):
    """
    Calculate temperature, concentration, and damage for each period.
    
    Parameters
    ----------
    m : ndarray
        Mitigation array
    tree : TreeModel
        Tree model
    damage : BPWDamage
        Damage model
    climate : BPWClimate
        Climate model
    emit_baseline : BPWEmissionBaseline
        Emission baseline model
    
    Returns
    -------
    tuple of ndarrays
        (exp_temp, exp_conc, exp_dam) - expected values per period
    """
    periods = tree.num_periods
    
    T_node = np.zeros(len(m))
    conc_node = np.zeros(len(m))
    dam_node = np.zeros(len(m))
    
    exp_temp = np.zeros(periods)
    exp_conc = np.zeros(periods)
    exp_dam = np.zeros(periods)
    
    for period in range(periods):
        nodes = tree.get_nodes_in_period(period)
        
        for node in range(nodes[0], nodes[1]+1):
            # Calculate damage
            dam_node[node] = damage._damage_function_node(m, node)
            
            # Calculate concentration
            conc_node[node] = climate.get_conc_at_node(m, node)
            
            # Calculate temperature
            mit_emit, _ = emit_baseline.get_mitigated_baseline(m, node=node, baseline='cumemit')
            T_node[node] = climate.TCRE_BEST_ESTIMATE * mit_emit[-1]
        
        # Take expectations over the period
        probs = tree.get_probs_in_period(period)
        exp_temp[period] = np.dot(T_node[nodes[0]:nodes[1]+1], probs)
        exp_conc[period] = np.dot(conc_node[nodes[0]:nodes[1]+1], probs)
        exp_dam[period] = np.dot(dam_node[nodes[0]:nodes[1]+1], probs)
    
    return exp_temp, exp_conc, exp_dam


def map_to_calendar_years(tree, period_values, target_years=COMMON_YEARS, start_year=START_YEAR):
    """
    Maps period-indexed values to calendar years on a common grid.
    
    Returns NaN for years not in this tree's decision times, or interpolates
    for years between decision points.
    
    Parameters
    ----------
    tree : TreeModel
        The tree model with decision_times attribute
    period_values : array-like
        Values indexed by period (length = tree.num_periods)
    target_years : list of int
        Calendar years to map to (default: COMMON_YEARS)
    start_year : int
        Starting calendar year (default: 2020)
    
    Returns
    -------
    np.ndarray
        Values mapped to target_years, with NaN for missing data
    """
    # Convert tree.decision_times to calendar years
    tree_years = [start_year + dt for dt in tree.decision_times]
    
    # period_values has length = num_periods, but tree_years has length = num_periods + 1
    # We need to ensure we don't index beyond period_values bounds
    num_periods = len(period_values)
    
    result = np.full(len(target_years), np.nan)
    
    for i, target_year in enumerate(target_years):
        if target_year in tree_years[:num_periods]:
            # Exact match - use the period value (only check first num_periods years)
            period_idx = tree_years.index(target_year)
            if period_idx < num_periods:
                result[i] = period_values[period_idx]
        elif target_year < tree_years[0] or target_year > tree_years[num_periods-1]:
            # Outside range of available data - keep as NaN
            continue
        else:
            # Between decision times - linear interpolation
            for j in range(num_periods - 1):
                if tree_years[j] < target_year < tree_years[j+1]:
                    weight = (target_year - tree_years[j]) / (tree_years[j+1] - tree_years[j])
                    result[i] = period_values[j] + weight * (period_values[j+1] - period_values[j])
                    break
    
    return result


def append_results_to_csv(results_dict, csv_path, max_retries=10, retry_delay=1.0): 
    # ConstraintAnalysis results are written in a joint CSV   
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    
    for attempt in range(max_retries):
        try:
            with open(csv_path, 'a', newline='') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    if not file_exists or os.path.getsize(csv_path) == 0:
                        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
                        writer.writeheader()
                    else:
                        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
                    
                    writer.writerow(results_dict)
                    f.flush()
                    os.fsync(f.fileno())
                    
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
            
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"Warning: Failed to write to CSV (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"ERROR: Failed to write to CSV after {max_retries} attempts: {e}")
                return False
    
    return False


def get_sample_filename():
    return os.path.join(DATA_DIR, f'LHC_samples_N{N_SAMPLES}_DIMS{DIMS}_ensemble_delayed.csv')


def generate_lhs_samples():
    samp_fname = get_sample_filename()
    
    print(f"\nGenerating {N_SAMPLES} Latin Hypercube Samples...")
    print(f"Parameter space dimension: {DIMS}")
    print(f"Parameter ranges:")
    for i, name in enumerate(param_names):
        print(f"  {name}: [{lbs[i]}, {ubs[i]}]")
    
    generate_samples(N_SAMPLES, DIMS, lbs, ubs, save_file=True, filename=samp_fname)
    
    print(f"Samples saved to: {samp_fname}\n")


def load_or_generate_lhs_samples():
    samp_fname = get_sample_filename()
    
    if not os.path.exists(samp_fname):
        print(f"\nSample file not found, generating new samples...")
        generate_lhs_samples()
    else:
        print(f"\nSample file found: {samp_fname}")
    
    param_vals = np.loadtxt(samp_fname, delimiter=',')
    print(f"Loaded {len(param_vals)} parameter samples")
    
    return param_vals


def get_cluster_config():    
    sge_task_id = os.environ.get('SGE_TASK_ID') # Get SGE task ID (1-indexed)
    if sge_task_id is None:
        print("ERROR: SGE_TASK_ID environment variable not found!")
        print("This script is designed to run as part of an SGE array job.")
        sys.exit(1)
    
    try:
        task_id = int(sge_task_id)
    except ValueError:
        print(f"ERROR: Invalid SGE_TASK_ID: {sge_task_id}")
        sys.exit(1)
    
    num_delays = len(delay_years)
    total_combinations = N_SAMPLES * num_delays
    
    task_index = task_id - 1
    
    if task_index >= total_combinations:
        print(f"Error: Task ID {task_id} exceeds total combinations ({total_combinations})\n")
        print(f"N_SAMPLES = {N_SAMPLES}")
        print(f"delay_years = {delay_years} (length {num_delays})")
        print(f"Expected task range: 1-{total_combinations}")
        sys.exit(1)
    
    # We iterate through delays for each sample:
    # task_index = sample_idx * num_delays + delay_idx
    sample_index = task_index // num_delays
    delay_idx = task_index % num_delays
    delay_year = delay_years[delay_idx]

    out_folder = os.environ.get('OUTPUT_FOLDER', output_folder)
    baseline = int(os.environ.get('BASELINE_NUM', baseline_num))
    sge_task_first = os.environ.get('SGE_TASK_FIRST', 'Unknown')
    sge_task_last = os.environ.get('SGE_TASK_LAST', 'Unknown')
    job_id = os.environ.get('JOB_ID', 'Unknown')
    
    print(f"\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total_combinations}")
    print(f"  Mapping: sample={sample_index}, delay_year={delay_year}")
    print(f"  Array range: {sge_task_first} to {sge_task_last}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline (SSP): {baseline}")
    print(f"\nConfiguration:")
    print(f"  Total samples: {N_SAMPLES}")
    print(f"  Delay years: {delay_years}")
    
    return sample_index, task_id, delay_year, out_folder, baseline


def setup_cluster_directories(out_folder):

    directories = [
        os.path.join(DATA_DIR, out_folder),
        os.path.join(DATA_DIR, out_folder, 'analysis'),
        os.path.join(DATA_DIR, out_folder, 'logs'),
        os.path.join(DATA_DIR, out_folder, 'samples')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("\nCreated directory structure:")
    for d in directories:
        print(f"  {d}")
    
    return directories


def run_ensemble_delayed_analysis(sample_index, delay_year, param_vals, 
                                  out_folder, baseline, test_mode, import_damages):
    
    ra, eis, tech_chg, tech_scale, pref, bs_premium, growth = param_vals[sample_index]
    
    name = f'sample{sample_index:04d}'
    
    print(f"\nSample {sample_index} | Delay Year: {delay_year}\n")
    
    print('\n**Model Parameters:')
    model_params = {
        'ra': ra,
        'eis': eis,
        'pref': pref,
        'growth': growth,
        'tech_chg': tech_chg,
        'tech_scale': tech_scale,
        'dam_func': dam_func,
        'baseline_num': baseline,
        'tip_on': tip_on,
        'bs_premium': bs_premium,
        'd_unc': d_unc,
        't_unc': t_unc,
        'no_free_lunch': no_free_lunch
    }
    pprint.pprint(model_params)
    
    if test_mode:
        print("\n***RUNNING IN TEST MODE***")
        N_generations_ga = 2
        N_iters_gs = 2
    else:
        print("\n***RUNNING IN FULL MODE***")
        N_generations_ga = 200
        N_iters_gs = 150
    
    print("\nInitializing model components...")
    
    # Tree model - baseline always uses standard decision times
    t_baseline = TreeModel(decision_times=[0, 10, 40, 80, 130, 180, 230],
                  prob_scale=1.0)
    
    # Delayed tree has second decision at delay_year instead of 10
    t_delay = TreeModel(decision_times=[0, delay_year, 40, 80, 130, 180, 230],
                  prob_scale=1.0)
    
    # Emission baseline model
    baseline_emission_model_baseline = BPWEmissionBaseline(tree=t_baseline,
                                                  baseline_num=baseline)
    baseline_emission_model_baseline.baseline_emission_setup()

    baseline_emission_model_delay = BPWEmissionBaseline(tree=t_delay,
                                                  baseline_num=baseline)
    baseline_emission_model_delay.baseline_emission_setup()
    
    # Climate class
    draws = 3 * 10**6
    climate_baseline = BPWClimate(
        t_baseline, baseline_emission_model_baseline, draws=draws
    )

    climate_delay = BPWClimate(
        t_delay, baseline_emission_model_delay, draws=draws
    )

    # Cost class
    c_baseline = BPWCost(t_baseline, emit_at_0=baseline_emission_model_baseline.baseline_gtco2[1],
                baseline_num=baseline, tech_const=tech_chg,
                tech_scale=tech_scale, cons_at_0=61880.0,
                backstop_premium=bs_premium, no_free_lunch=no_free_lunch)
    
    c_delay = BPWCost(t_delay, emit_at_0=baseline_emission_model_delay.baseline_gtco2[1],
                baseline_num=baseline, tech_const=tech_chg,
                tech_scale=tech_scale, cons_at_0=61880.0,
                backstop_premium=bs_premium, no_free_lunch=no_free_lunch)
    
    # Damage class
    d_m = 0.1
    mitigation_constants = np.arange(0, 1 + d_m, d_m)[::-1]
    df_baseline = BPWDamage(tree=t_baseline, emit_baseline=baseline_emission_model_baseline,
                   climate=climate_baseline, mitigation_constants=mitigation_constants,
                   draws=draws)

    df_delay = BPWDamage(tree=t_delay, emit_baseline=baseline_emission_model_delay,
                   climate=climate_delay, mitigation_constants=mitigation_constants,
                   draws=draws)


    damsim_filename_baseline = ''.join(["simulated_damages_df", str(dam_func),
                               "_TP", str(tip_on), "_SSP", str(baseline),
                               "_dunc", str(d_unc), "_tunc", str(t_unc)])
    
    damsim_filename_delay = ''.join(["simulated_damages_df", str(dam_func),
                               "_TP", str(tip_on), "_SSP", str(baseline),
                               "_dunc", str(d_unc), "_tunc", str(t_unc)])
    
    print(f"Damage simulation: {damsim_filename_baseline}")
    
    if import_damages:
        try:
            df_baseline.import_damages(file_name=damsim_filename_baseline)
            df_delay.import_damages(file_name=damsim_filename_delay)
            print("Successfully imported damage simulation\n")
        except Exception as e:
            print(f"Warning: Could not import damages ({e})")
            print("Running damage simulation...")
            df_baseline.damage_simulation(filename=damsim_filename_baseline, save_simulation=True,
                                 dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                                 t_unc=t_unc)
            df_delay.damage_simulation(filename=damsim_filename_delay, save_simulation=True,
                                 dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                                 t_unc=t_unc)
    else:
        print("Running damage simulation...")
        df_baseline.damage_simulation(filename=damsim_filename_baseline, save_simulation=True,
                             dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                             t_unc=t_unc)
        df_delay.damage_simulation(filename=damsim_filename_delay, save_simulation=True,
                             dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                             t_unc=t_unc)

    u_baseline = EZUtility(tree=t_baseline, damage=df_baseline, cost=c_baseline, period_len=5.0, eis=eis, ra=ra,
                  time_pref=pref, cons_growth=growth)
    
    u_delay = EZUtility(tree=t_delay, damage=df_delay, cost=c_delay, period_len=5.0, eis=eis, ra=ra,
                  time_pref=pref, cons_growth=growth)

    print("Model components initialized\n")
    
    
    print("\nRUNNING OPTIMAL (UNCONSTRAINED) SCENARIO\n")
    
    # no constraints for optimal scenario
    fixed_indices_opt = None
    fixed_values_opt = None
    
    ga_model_opt = GeneticAlgorithm(
        pop_amount=400,
        num_generations=N_generations_ga,
        cx_prob=0.8, 
        mut_prob=0.50, 
        bound=1.5,
        num_feature=t_baseline.num_decision_nodes,
        utility=u_baseline, 
        fixed_values=fixed_values_opt,
        fixed_indices=fixed_indices_opt,
        print_progress=True
    )
    
    gs_model_opt = GradientSearch(
        var_nums=t_baseline.num_decision_nodes,  
        utility=u_baseline, 
        accuracy=5.e-7,
        iterations=N_iters_gs,
        fixed_values=fixed_values_opt,
        fixed_indices=fixed_indices_opt,
        print_progress=True
    )
    
    print("Running Genetic Algorithm (optimal)...")
    final_pop_opt, fitness_opt = ga_model_opt.run()
    sort_pop_opt = final_pop_opt[np.argsort(fitness_opt)][::-1]
    
    print("Running Gradient Search (optimal)...")
    m_optimal, u_optimal = gs_model_opt.run(initial_point_list=sort_pop_opt, topk=1)
    
    print(f"\nOptimal scenario complete:")
    print(f"  First-period mitigation:  {m_optimal[0]:.6f}")
    print(f"  Carbon price:             ${c_baseline.price(0, m_optimal[0], 0):.2f} per ton")
    print(f"  Utility:                  {u_optimal:.10f}\n")
    
    # Calculate climate output for timeseries (but don't save individual files)
    co_opt = ClimateOutput(u_baseline)
    co_opt.calculate_output(m_optimal)
    
    print(f"\nRUNNING DELAYED ACTION SCENARIO (DELAY={delay_year} YEARS)\n")
    
    fixed_indices_delay = get_delay_nodes(t_delay, 1)
    fixed_values_delay = np.zeros(len(fixed_indices_delay))
    
    print(f"Constraint configuration:")
    print(f"  Total decision nodes:        {t_delay.num_decision_nodes}")
    print(f"  Number of nodes constrained: {len(fixed_indices_delay)}")
    print(f"  Constrained node indices:    {fixed_indices_delay}\n")
    
    ga_model_delay = GeneticAlgorithm(
        pop_amount=400,
        num_generations=N_generations_ga,
        cx_prob=0.8, 
        mut_prob=0.50, 
        bound=1.5,
        num_feature=t_delay.num_decision_nodes,
        utility=u_delay, 
        fixed_values=fixed_values_delay,
        fixed_indices=fixed_indices_delay,
        print_progress=True
    )
    
    gs_model_delay = GradientSearch(
        var_nums=t_delay.num_decision_nodes,
        utility=u_delay, 
        accuracy=5.e-7,
        iterations=N_iters_gs,
        fixed_values=fixed_values_delay,
        fixed_indices=fixed_indices_delay,
        print_progress=True
    )
    
    print("Running Genetic Algorithm (delayed)...")
    final_pop_delay, fitness_delay = ga_model_delay.run()
    sort_pop_delay = final_pop_delay[np.argsort(fitness_delay)][::-1]
    
    print("Running Gradient Search (delayed)...")
    m_delayed, u_delayed = gs_model_delay.run(initial_point_list=sort_pop_delay, topk=1)
    
    for idx in fixed_indices_delay:
        if abs(m_delayed[idx]) > 1e-10:
            print(f"Warning: Constrained node {idx} not zero: m_delayed[{idx}]={m_delayed[idx]:.10f}")
    
    print(f"\nDelayed scenario complete:")
    print(f"  First-period mitigation:  {m_delayed[0]:.6f} (constrained to 0)")
    print(f"  Carbon price:             ${c_delay.price(0, m_delayed[0], 0):.2f} per ton")
    print(f"  Utility:                  {u_delayed:.10f}\n")
    
    # Calculate climate output for timeseries (but don't save individual files)
    co_delay = ClimateOutput(u_delay)
    co_delay.calculate_output(m_delayed)
    
    print("\nCONSTRAINT ANALYSIS (COMPARING OPTIMAL VS DELAYED)\n")
    
    ca = ConstraintAnalysis(u_delay, u_baseline, m_delayed, m_optimal)
    
    print(f"\nCOMPARISON SUMMARY (SAMPLE={sample_index}, DELAY={delay_year})\n")
    print(f"\nOptimization Results:")
    print(f"  Optimal first-period mitigation:   {m_optimal[0]:.6f}")
    print(f"  Delayed first-period mitigation:   {m_delayed[0]:.6f}")
    print(f"  Mitigation foregone:               {m_optimal[0] - m_delayed[0]:.6f}")
    
    print(f"\nUtility Comparison:")
    print(f"  Optimal utility:                   {u_optimal:.10f}")
    print(f"  Delayed utility:                   {u_delayed:.10f}")
    print(f"  Utility loss:                      {ca.con_cost:.10f}")
    print(f"  Relative loss:                     {(ca.con_cost/u_optimal)*100:.4f}%")
    
    print(f"\nEconomic Impacts:")
    print(f"  Consumption compensation (abs):    {ca.delta_c:.6f}")
    print(f"  Compensation (% of year 0 cons):   {ca.delta_c_pct:.4f}%")
    print(f"  Compensation (billions $):         ${ca.delta_c_billions:.2f}B")
    print(f"  Emission reduction foregone:       {ca.delta_emission_gton:.4f} Gt CO2")
    
    if ca.deadweight is not None:
        print(f"\nDeadweight Analysis:")
        print(f"  Deadweight cost:                   ${ca.deadweight:.2f} per ton CO2")
    
    results_dict = {
        # Run identifiers
        'sample_index': sample_index,
        'delay_year': delay_year,
        'task_id': os.environ.get('SGE_TASK_ID', 'unknown'),
        
        # Parameter values (from LHS)
        'ra': float(ra),
        'eis': float(eis),
        'pref': float(pref),
        'tech_chg': float(tech_chg),
        'tech_scale': float(tech_scale),
        'bs_premium': float(bs_premium),
        'growth': float(growth),
        
        # Fixed parameters
        'baseline_num': int(baseline),
        'dam_func': int(dam_func),
        'tip_on': int(tip_on),
        'd_unc': int(d_unc),
        't_unc': int(t_unc),
        'no_free_lunch': bool(no_free_lunch),
        
        # Mitigation levels
        'm_optimal_period0': float(m_optimal[0]),
        'm_delayed_period0': float(m_delayed[0]),
        'mitigation_foregone': float(m_optimal[0] - m_delayed[0]),
        
        # Utility metrics
        'u_optimal': float(u_optimal),
        'u_delayed': float(u_delayed),
        'utility_loss': float(ca.con_cost),
        'utility_loss_pct': float((ca.con_cost/u_optimal)*100) if u_optimal != 0 else np.nan,
        
        # Economic impacts
        'delta_c': float(ca.delta_c) if ca.delta_c is not None else np.nan,
        'delta_c_pct': float(ca.delta_c_pct) if ca.delta_c_pct is not None else np.nan,
        'delta_c_billions': float(ca.delta_c_billions) if ca.delta_c_billions is not None else np.nan,
        'year0_cons_delayed': float(ca.year0_cons_delayed),
        'delta_emission_gton': float(ca.delta_emission_gton),
        'deadweight_per_ton': float(ca.deadweight) if ca.deadweight is not None else np.nan,
        
        # Carbon prices
        'carbon_price_delayed': float(c_delay.price(0, m_delayed[0], 0)),
        'carbon_price_optimal': float(c_baseline.price(0, m_optimal[0], 0)),
    }
    
    csv_path = os.path.join(DATA_DIR, out_folder, 'analysis', f'{out_folder}_consolidated_results.csv')
    
    print(f"Appending results to: {csv_path}")
    success = append_results_to_csv(results_dict, csv_path)
    
    if success:
        print(f"Successfully appended results to consolidated CSV\n")
    else:
        print(f"Warning: Could not append to consolidated CSV (individual files still saved)\n")
    
    # Build timeseries data on common temporal grid
    print("\nCalculating climate metrics for timeseries...")
    
    # Calculate temperature, concentration, and damage for both scenarios
    exp_temp_opt, exp_conc_opt, exp_dam_opt = calculate_period_climate_metrics(
        m_optimal, t_baseline, df_baseline, climate_baseline, baseline_emission_model_baseline
    )
    
    exp_temp_delay, exp_conc_delay, exp_dam_delay = calculate_period_climate_metrics(
        m_delayed, t_delay, df_delay, climate_delay, baseline_emission_model_delay
    )
    
    print("Mapping timeseries data to common temporal grid...")
    print(f"  Common years: {COMMON_YEARS}")
    
    # Map optimal scenario to common grid
    m_opt_mapped = map_to_calendar_years(t_baseline, co_opt.expected_period_mitigation, COMMON_YEARS)
    T_opt_mapped = map_to_calendar_years(t_baseline, exp_temp_opt, COMMON_YEARS)
    conc_opt_mapped = map_to_calendar_years(t_baseline, exp_conc_opt, COMMON_YEARS)
    dam_opt_mapped = map_to_calendar_years(t_baseline, exp_dam_opt, COMMON_YEARS)
    price_opt_mapped = map_to_calendar_years(t_baseline, co_opt.expected_period_price, COMMON_YEARS)
    
    # Map delayed scenario to common grid
    m_delay_mapped = map_to_calendar_years(t_delay, co_delay.expected_period_mitigation, COMMON_YEARS)
    T_delay_mapped = map_to_calendar_years(t_delay, exp_temp_delay, COMMON_YEARS)
    conc_delay_mapped = map_to_calendar_years(t_delay, exp_conc_delay, COMMON_YEARS)
    dam_delay_mapped = map_to_calendar_years(t_delay, exp_dam_delay, COMMON_YEARS)
    price_delay_mapped = map_to_calendar_years(t_delay, co_delay.expected_period_price, COMMON_YEARS)
    
    # Build timeseries dictionary
    timeseries_dict = {
        # Run identifiers
        'sample_index': sample_index,
        'delay_year': delay_year,
        'task_id': os.environ.get('SGE_TASK_ID', 'unknown'),
        
        # Parameter values (from LHS)
        'ra': float(ra),
        'eis': float(eis),
        'pref': float(pref),
        'tech_chg': float(tech_chg),
        'tech_scale': float(tech_scale),
        'bs_premium': float(bs_premium),
        'growth': float(growth),
        
        # Fixed parameters
        'baseline_num': int(baseline),
        'dam_func': int(dam_func),
        'tip_on': int(tip_on),
        'd_unc': int(d_unc),
        't_unc': int(t_unc),
        'no_free_lunch': bool(no_free_lunch),
        
        # Summary metrics
        'u_optimal': float(u_optimal),
        'u_delayed': float(u_delayed),
        'utility_loss': float(ca.con_cost),
    }
    
    # Add timeseries organized by variable type (mitigation, temperature, concentration, damage, price)
    # For each variable, add optimal years first, then delayed years
    
    # Mitigation timeseries
    for i, year in enumerate(COMMON_YEARS):
        timeseries_dict[f'm_opt_{year}'] = float(m_opt_mapped[i]) if not np.isnan(m_opt_mapped[i]) else np.nan
    for i, year in enumerate(COMMON_YEARS):
        timeseries_dict[f'm_delay_{year}'] = float(m_delay_mapped[i]) if not np.isnan(m_delay_mapped[i]) else np.nan
    
    # Temperature timeseries
    for i, year in enumerate(COMMON_YEARS):
        timeseries_dict[f'T_opt_{year}'] = float(T_opt_mapped[i]) if not np.isnan(T_opt_mapped[i]) else np.nan
    for i, year in enumerate(COMMON_YEARS):
        timeseries_dict[f'T_delay_{year}'] = float(T_delay_mapped[i]) if not np.isnan(T_delay_mapped[i]) else np.nan
    
    # Concentration timeseries
    for i, year in enumerate(COMMON_YEARS):
        timeseries_dict[f'conc_opt_{year}'] = float(conc_opt_mapped[i]) if not np.isnan(conc_opt_mapped[i]) else np.nan
    for i, year in enumerate(COMMON_YEARS):
        timeseries_dict[f'conc_delay_{year}'] = float(conc_delay_mapped[i]) if not np.isnan(conc_delay_mapped[i]) else np.nan
    
    # Damage timeseries
    for i, year in enumerate(COMMON_YEARS):
        timeseries_dict[f'dam_opt_{year}'] = float(dam_opt_mapped[i]) if not np.isnan(dam_opt_mapped[i]) else np.nan
    for i, year in enumerate(COMMON_YEARS):
        timeseries_dict[f'dam_delay_{year}'] = float(dam_delay_mapped[i]) if not np.isnan(dam_delay_mapped[i]) else np.nan
    
    # Carbon price timeseries
    for i, year in enumerate(COMMON_YEARS):
        timeseries_dict[f'price_opt_{year}'] = float(price_opt_mapped[i]) if not np.isnan(price_opt_mapped[i]) else np.nan
    for i, year in enumerate(COMMON_YEARS):
        timeseries_dict[f'price_delay_{year}'] = float(price_delay_mapped[i]) if not np.isnan(price_delay_mapped[i]) else np.nan
    
    # Save timeseries to consolidated CSV
    timeseries_csv_path = os.path.join(DATA_DIR, out_folder, 'analysis', f'{out_folder}_consolidated_timeseries.csv')
    
    print(f"Appending timeseries to: {timeseries_csv_path}")
    success_ts = append_results_to_csv(timeseries_dict, timeseries_csv_path)
    
    if success_ts:
        print(f"Successfully appended timeseries to consolidated CSV\n")
    else:
        print(f"Warning: Could not append timeseries to consolidated CSV\n")


def main():    
    print("\nCAP6 ENSEMBLE DELAYED ACTION ANALYSIS - CLUSTER ARRAY JOB\n")

    sample_index, task_id, delay_year, out_folder, baseline = get_cluster_config()

    setup_cluster_directories(out_folder)
    
    param_vals = load_or_generate_lhs_samples()
    
    if sample_index >= len(param_vals):
        print(f"ERROR: Sample index {sample_index} exceeds available samples ({len(param_vals)})")
        sys.exit(1)
    
    if task_id == 1:
        samples_copy = os.path.join(DATA_DIR, out_folder, 'samples', 
                                    f'LHC_samples_N{N_SAMPLES}_DIMS{DIMS}.csv')
        np.savetxt(samples_copy, param_vals, delimiter=',')
        print(f"\nSaved copy of samples to: {samples_copy}")
    

    print(f"RUNNING: Sample {sample_index}/{len(param_vals)-1}")
    print(f"DELAY YEAR: {delay_year}")
    print(f"TASK: {task_id}\n")
    
    print(f"\nExecution Configuration:")
    print(f"  Test mode:       {test_mode}")
    print(f"  Import damages:  {import_damages}")
    print(f"  Baseline (SSP):  {baseline}")
    
    try:
        run_ensemble_delayed_analysis(sample_index, delay_year, param_vals,
                                     out_folder, baseline, test_mode, import_damages)
    except Exception as e:
        print(f"ERROR running sample {sample_index} with delay {delay_year}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\nTASK COMPLETE: Sample {sample_index} (delay {delay_year})")
    print(f"Task ID: {task_id}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
