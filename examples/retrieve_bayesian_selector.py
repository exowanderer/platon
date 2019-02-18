from __future__ import print_function

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from multiprocessing import Pool, cpu_count

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

try:
    import nestle
except:
    !pip install nestle
    import nestle

try:
    import corner
except:
    !pip install corner
    import corner

try:
    from pygtc import plotGTC
except:
    !pip install pygtc
    from pygtc import plotGTC

try:
    import platon
    del platon
except:
    !pip install git+https://github.com/ideasrule/platon

from datetime import datetime

from pandas import DataFrame
from platon.fit_info import FitInfo
from platon.retriever import Retriever
from platon.constants import R_sun, R_jup, M_jup
from platon.constants import METRES_TO_UM
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.errors import AtmosphereError

from sklearn.externals import joblib

from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument('-bm', '--bayesianmodel', required=False, default='emcee', type=str, help="type of bayesian model: 'mutlinest' or 'emcee'")
ap.add_argument('-nw', '--num_walkers', required=False, default=50, type=int, help='number of walkers in emcee')
ap.add_argument('-ns', '--num_steps', required=False, default=50, type=int, help='number of steps in emcee')

args = vars(ap.parse_args())

bayesian_model = args['bayesianmodel'] if 'bayesianmodel' in args.keys() else ap.get_default('bayesianmodel')
nwalkers = args['num_walkers'] if 'num_walkers' in args.keys() else ap.get_default('num_walkers')
nsteps = args['num_steps'] if 'num_steps' in args.keys() else ap.get_default('num_steps')

planet_name = 'HD 209458 b'
hd209458b = exoMAST_API(planet_name)
hd209458b.get_spectra()

wavelengths, wave_errs, depths, errors = hd209458b.planetary_spectra_table.values.T

wavelengths = (wavelengths*micron).to(meter).value
wave_errs = (wave_errs*micron).to(meter).value

wave_bins = np.transpose([wavelengths - wave_errs, wavelengths + wave_errs])

#create a Retriever object
retriever = Retriever()

R_guess = 1.4 * R_jup
T_guess = 1200

Rs = 1.19 * R_sun
Mp = 0.73 * M_jup
Rp = R_guess
T_eq = T_guess
logZ = 0
CO_ratio = 0.53
log_cloudtop_P = 4
log_scatt_factor = 0
scatt_slope = 4
error_multiple = 1
T_star = 6091

#create a FitInfo object and set best guess parameters
fit_info = retriever.get_default_fit_info(
                        Rs=Rs, Mp=Mp, Rp=Rp, T=T_eq,
                        logZ=logZ, CO_ratio=CO_ratio, 
                        log_cloudtop_P=log_cloudtop_P,
                        log_scatt_factor=log_scatt_factor, 
                        scatt_slope=scatt_slope, 
                        error_multiple=error_multiple, 
                        T_star=T_star)

#Add fitting parameters - this specifies which parameters you want to fit
#e.g. since we have not included cloudtop_P, it will be fixed at the value specified in the constructor
fit_info.add_gaussian_fit_param('Rs', 0.02*R_sun)
fit_info.add_gaussian_fit_param('Mp', 0.04*M_jup)

fit_info.add_uniform_fit_param('Rp', 0.9*R_guess, 1.1*R_guess)
fit_info.add_uniform_fit_param('T', 0.5*T_guess, 1.5*T_guess)
fit_info.add_uniform_fit_param("log_scatt_factor", 0, 1)
fit_info.add_uniform_fit_param("logZ", -1, 3)
fit_info.add_uniform_fit_param("log_cloudtop_P", -0.99, 5)
fit_info.add_uniform_fit_param("error_multiple", 0.5, 5)

#Use Nested Sampling to do the fitting
# with ThreadPoolExecutor() as executor:
# with ProcessPoolExecutor() as executor:
time_stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

if bayesian_model == 'multinest':
    with Pool(cpu_count()) as executor:
        result = retriever.run_multinest(wave_bins, depths, errors, fit_info, nestle_kwargs={'pool':executor})#, 'bootstrap':0 # bootstrap for `dynesty`
    
    result_dict = {'samples':result.samples, 'weights':result.weights, 'logl':result.logl}
    joblib.dump(result_dict, 'multinest_results_{}.joblib.save'.format(time_stamp))
elif bayesian_model == 'emcee':
    result = retriever.run_emcee(wave_bins, depths, errors, fit_info, nwalkers=nwalkers, nsteps=nsteps)
    
    result_dict = {'flatchain':result.flatchain, 'flatlnprob':result.flatlnprobability, 'chain':result.chain, 'lnprob':result.lnprobability}
    joblib.dump(result_dict, 'emcee_results_{}walkers_{}steps_{}.joblib.save'.format(nwalkers, nsteps, time_stamp))
else:
    raise ValueError("Options for `bayesian_model` (-bm, --bayesianmodel) must be either 'multinest' or 'emcee'")

# Establish the Range in Wavelength to plot high resolution figures
wave_min = wave_bins.min()
wave_max = wave_bins.max()

n_theory_pts = 500
wavelengths_theory = np.linspace(wave_min, wave_max, n_theory_pts)
half_diff_lam = 0.5*np.median(np.diff(wavelengths_theory))

# Setup calculator to use the theoretical wavelengths
calculator = TransitDepthCalculator(include_condensation=True)
calculator.change_wavelength_bins(np.transpose([wavelengths_theory-half_diff_lam, wavelengths_theory+half_diff_lam]))

retriever._validate_params(fit_info, calculator)

# Allocate the best-fit parameters from the `result` class
if bayesian_model == 'multinest':
    best_params_arr = result.samples[np.argmax(result.logl)]
elif bayesian_model == 'emcee':
    best_params_arr = result.flatchain[np.argmax(result.flatlnprobability)]
else:
    raise ValueError("Options for `bayesian_model` (-bm, --bayesianmodel) must be either 'multinest' or 'emcee'")

best_params_dict = {key:val for key,val in zip(fit_info.fit_param_names, best_params_arr)}

# Set the static parameteres to the default values
for key in fit_info.all_params.keys():
    if key not in best_params_dict.keys():
        best_params_dict[key] = fit_info.all_params[key].best_guess

# Assign the best fit model parameters to necessary variables
Rs = best_params_dict['Rs']
Mp = best_params_dict['Mp']
Rp = best_params_dict['Rp']
T_eq = best_params_dict['T']
logZ = best_params_dict['logZ']
CO_ratio = best_params_dict['CO_ratio']
log_cloudtop_P = best_params_dict['log_cloudtop_P']
log_scatt_factor = best_params_dict['log_scatt_factor']
scatt_slope = best_params_dict['scatt_slope']
error_multiple = best_params_dict['error_multiple']
T_star = best_params_dict['T_star']

T_spot = best_params_dict['T_spot']
spot_cov_frac = best_params_dict['spot_cov_frac']
frac_scale_height = best_params_dict['frac_scale_height']
log_number_density = best_params_dict['log_number_density']
log_part_size = best_params_dict['log_part_size']
part_size_std = best_params_dict['part_size_std']
ri = best_params_dict['ri']

# Compute best-fit theoretical model
try:
    wavelengths, calculated_depths = calculator.compute_depths(
        Rs, Mp, Rp, T_eq, logZ, CO_ratio,
        scattering_factor=10**log_scatt_factor, scattering_slope=scatt_slope,
        cloudtop_pressure=10**log_cloudtop_P, T_star=T_star)
except AtmosphereError as e:
    print(e)

# Plot the data on top of the best fit high-resolution model
plt.errorbar(METRES_TO_UM * np.mean(wave_bins, axis=1), depths, yerr=errors, fmt='.', color='k', zorder=100)
plt.plot(METRES_TO_UM * wavelengths, calculated_depths)

plt.xlabel("Wavelength (um)")
plt.ylabel("Transit depth")
plt.xscale('log')

plt.tight_layout()

if bayesian_model == 'multinest':
    plt.savefig('multinest_best_fit_{}.png'.format(time_stamp))
elif bayesian_model == 'emcee':
    plt.savefig('emcee_best_fit_{}walkers_{}steps_{}.png'.format(nwalkers, nsteps, time_stamp))
else:
    raise ValueError("Options for `bayesian_model` (-bm, --bayesianmodel) must be either 'multinest' or 'emcee'")


# GTC: Grand Triangle of Confusion; i.e. Prettier Corner Plot
if bayesian_model == 'multinest':
    res_flatchain_df = DataFrame(result.samples, columns=fit_info.fit_param_names)
elif bayesian_model == 'emcee':
    res_flatchain_df = DataFrame(result.flatchain, columns=fit_info.fit_param_names)
else:
    raise ValueError("Options for `bayesian_model` (-bm, --bayesianmodel) must be either 'multinest' or 'emcee'")

if 'Rs' in fit_info.fit_param_names: res_flatchain_df['Rs'] = res_flatchain_df['Rs'] / R_sun
if 'Rp' in fit_info.fit_param_names: res_flatchain_df['Rp'] = res_flatchain_df['Rp'] / R_jup
if 'Mp' in fit_info.fit_param_names: res_flatchain_df['Mp'] = res_flatchain_df['Mp'] / M_jup

plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10,10)

fit_param_names = ['Rs [Rsun]', 'Mp [Mjup]', 'Rp [Rjup]', 'T [K]', 'log(haze)', 'logZ [FeH]', 'log(P_c) [mbar]', 'err_mod']

pygtc_out = plotGTC(res_flatchain_df.values,
                    weights=result.weights if bayesian_model == 'multimodel' else None,
                    nContourLevels=3, 
                    paramNames=fit_param_names,
                    # range=[0.99] * result.flatchain.shape[1],
                    # chainLabels=['Not Cheap', 'Cheap'],
                    customLabelFont={'size':10},
                    customTickFont={'size':7},
                    # customLegendFont={'size':30},
                    figureSize=12);

if bayesian_model == 'multinest':
    plt.savefig('multinest_gtc_{}.png'.format(time_stamp))
elif bayesian_model == 'emcee':
    plt.savefig('emcee_gtc_{}walkers_{}steps_{}.png'.format(nwalkers, nsteps, time_stamp))
else:
    raise ValueError("Options for `bayesian_model` (-bm, --bayesianmodel) must be either 'multinest' or 'emcee'")
