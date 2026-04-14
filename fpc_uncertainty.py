import numpy as np
from pyspedas import get_data, tinterpol, store_data

def fpc_uncertainty(dist, 
                    eigen, 
                    vth,
                    ve0,
                    edges, 
                    spacecraft_id=1, 
                    species='electron',
                    counts_to_mask=0,
                    ecut=None):

  ''' 
  Propagates Poisson counting uncertainties through the FPC pipeline
  following Afshari et al. Table B1 and Appendix B https://doi.org/10.1029/2021JA029578.

  The electric field is treated as a constant consistent with Afshari et al. All uncertainty
  propagates from the Poisson counting statistics reported in the FPI level 2 disterr variable.

  Parameters
  ----------
  dist : list of dict
    Particle distribution dicts, each with keys:
    'start_time', 'end_time', 'energy', 'theta', 'phi', 'data'.
  eigen : list of ndarray
    FAC basis vectors [e_par, e_perp1, e_perp2], each shape (3,).
    Must be the same eigen used in the FPC computation.
  vth : float
    Thermal velocity in m/s. Must be the same vth used in the FPC
    computation
  ve0 : list of int
    Mean bulk velocity values in m/s for x,y,z.
  edges : dict
    Velocity grid edges from field_particle_correlation, containing
    keys 'vpar' and 'vperp'.
  spacecraft_id : int, optional
    MMS spacecraft ID (1-4). Default is 1.
  species : str, optional
    Particle species, 'ion' or 'electron'. Default is 'electron'
  counts_to_mask : int, optional
    Minimum bin count threshold. Default is 0.

  Returns
  -------
  uncertainty : dict
    Uncertainty dicts, with keys sigma(1-7) containing all uncertainties
    outlined in Table B1 of Afshari et al.

  Raises
  ------
  ValueError
     If required tplot variables cannot be retreived, disterr is
    unavailable, or the pre-processed E field has not been stored

  Notes
  -----
  Requires field_particle_correlation to have been run first so that
  'e_field_transformed_smooth' exists in the tplot store.
  Bin sizes follow Afshari et al. : dv_perp = dv_par = 0.1 v_th,e/i
  '''

  if species == 'electron':
    q = -1.60217663e-19
    mass = 9.10938371e-31
    disterr_tvar = f'mms{spacecraft_id}_des_disterr_brst'
  else: 
    q = 1.60217663e-19
    mass = 1.67262192e-27
    disterr_tvar = f'mms{spacecraft_id}_dis_disterr_brst'

  vpar_edges    = edges['vpar']
  vperp_edges   = edges['vperp']
  vperp_1_edges = edges['vperp1']
  vperp_2_edges = edges['vperp2']
                      
  nbins_par   = len(vpar_edges) - 1
  nbins_perp  = len(vperp_edges) - 1
  nbins_perp1 = len(vperp_1_edges) - 1
  nbins_perp2 = len(vperp_2_edges) - 1

  vpar_centres    =  0.5 * (vpar_edges[:-1]  + vpar_edges[1:])
  vperp_centres   = 0.5 * (vperp_edges[:-1] + vperp_edges[1:])
  vperp_1_centres = 0.5 * (vperp_1_edges[:-1] + vperp_1_edges[1:])
  vperp_2_centres = 0.5 * (vperp_2_edges[:-1] + vperp_2_edges[1:])
                      
  dvpar       = (vpar_edges[1]  - vpar_edges[0])  * vth
  dvperp      = (vperp_edges[1] - vperp_edges[0]) * vth
  dvperp1     = (vperp_1_edges[1] - vperp_1_edges[0]) * vth
  dvperp2     = (vperp_2_edges[1] - vperp_2_edges[0]) * vth
                      
  vpar_phys     = vpar_centres  * vth
  vperp_phys    = vperp_centres * vth
  vperp_1_phys  = vperp_1_centres * vth
  vperp_2_phys  = vperp_2_centres * vth
                      
  # ================================================================
  # LOAD DISTERR - sigma = f / sqrt(N), provided directly by FPI L2
  # ================================================================

  disterr_data = get_data(disterr_tvar)
  if disterr_data is None:
    raise ValueError(
      f"Could not retrieve '{disterr_tvar}'. "
      "Ensure FPI burst data has been loaded"
    )

  # ================================================================
  # LOAD E FIELD - treated as a constant, used only as a multiplier
  # Retrived from tplot store set by field_particle_correlation
  # ================================================================

  e_data = get_data('e_field_transformed_smooth')
  if e_data is None:
    raise ValueError(
      "Could not retrieve 'e_field_transformed_smooth'. "
      "Ensure field_particle_correlation has been run first"
    )

  # ================================================================
  # DISTRIBUTION TIMES AND VELOCITY GRID
  # ================================================================

  t_dist = np.array(
                [(d['start_time'] + d['end_time']) * 0.5 for d in dist], dtype=float
  )
  n_t = len(t_dist)

  energy = np.array([d['energy'] for d in dist], dtype=float)
  theta  = np.deg2rad(np.array([dist[0]['theta']], dtype=float))
  phi    = np.deg2rad(np.array([dist[0]['phi']],   dtype=float))

  # --- Optional photoelectron cut ---
  if ecut is not None:
    energy = np.where(energy > ecut, energy, np.nan)

  energy = np.where(energy > 0, energy, 1e-12)

  v = np.sqrt(2 * energy * np.abs(q) / mass)

  vx    = v * np.cos(theta[0]) * np.cos(phi[0]) - ve0[0]
  vy    = v * np.cos(theta[0]) * np.sin(phi[0]) - ve0[1]#
  vz    = v * np.sin(theta[0]) - ve0[2]
  vvec  = np.stack([vx, vy, vz], axis=-1)

  vpar    = np.tensordot(vvec, eigen[0], axes=([-1], [0]))
  vperp_1 = np.tensordot(vvec, eigen[1], axes=([-1], [0]))
  vperp_2 = np.tensordot(vvec, eigen[2], axes=([-1], [0]))
  vperp   = np.sqrt(vperp_1**2 + vperp_2**2)

  vpar_mean    = np.nanmean(vpar, axis=0)    # (N_energy, N_theta, N_phi)
  vperp_1_mean = np.nanmean(vperp_1, axis=0)
  vperp_2_mean = np.nanmean(vperp_2, axis=0)

  vpar_n  = vpar_mean.ravel() / vth
  vperp_n = np.nanmean(vperp, axis=0).ravel() / vth

  # ================================================================
  # INTERPOLATE DISTERR AND E FIELD ONTO DISTRIBUTION TIMES
  # ================================================================  

  store_data('_disterr_full', data={'x': disterr_data.times, 'y': disterr_data.y})
  tinterpol('_disterr_full', t_dist, newname='_disterr_interp')
  disterr_interp = get_data('_disterr_interp')
  if disterr_interp is None:
    raise ValueError("Could not interpolate disterr onto distribution times.")

  store_data('_e_unc_full', data={'x': e_data.times, 'y': e_data.y})
  tinterpol('_e_unc_full', t_dist, newname='_e_unc_interp')
  e_interp = get_data('_e_unc_interp')
  if e_interp is None:
    raise ValueError("Could not interpolate E field onto distribution times.")

  # Project E field onto FAC - shape (N_t,)
  epar = np.dot(e_interp.y * 1e-3, eigen[0])
  eperp_1 = np.dot(e_interp.y * 1e-3, eigen[1])                    
  eperp_2 = np.dot(e_interp.y * 1e-3, eigen[2])

  # ================================================================
  # sigma: original data uncertainty from disterr
  # Units : s^3/cm^-6 -> s^3/m^-6
  # Shape : (N_t, N_energy, N_theta, N_phi)
  # ================================================================  

  sigma = disterr_interp.y * 1e12
  sigma = sigma.transpose(0, 1, 3, 2) # reorder to (N_t, N_energy, N_theta, N_phi)

  
  # ================================================================
  # sigma_1: background distribution uncertainty
  # sigma' = (1/n) * sqrt(sum_j sigma_j^2) over time axis
  # Shape : (N_energy, N_theta, N_phi)
  # ================================================================    

  sigma_1 = (1 / n_t) * np.sqrt(np.nansum(sigma**2, axis=0))

  # ================================================================
  # sigma_2: fluctuation uncertainty
  # sigma'' = sqrt(sigma'^2 + sigma^2)
  # Shape : (N_t, N_energy, N_theta, N_phi)
  # ================================================================

  sigma_2 = np.sqrt(sigma_1[np.newaxis, :]**2 + sigma**2)

  # ================================================================
  # sigma_3: correlation uncertainty at each time step, E field
  #          treated as exact.
  # sigma''' = |q * v * sigma'' * E|
  # Shape : (N_t, N_energy, N_theta, N_phi)
  # ================================================================  

  sigma_3_par = np.abs(
    q * vpar_mean[np.newaxis, :]
    * epar[:, np.newaxis, np.newaxis, np.newaxis]
    * sigma_2
  )

  sigma_3_perp_1 = np.abs(
    q * vperp_1_mean[np.newaxis, :]
    * eperp_1[:, np.newaxis, np.newaxis, np.newaxis]
    * sigma_2
  )

  sigma_3_perp_2 = np.abs(
    q * vperp_2_mean[np.newaxis, :]
    * eperp_2[:, np.newaxis, np.newaxis, np.newaxis]
    * sigma_2
  )

  sigma_3 = {
    'par'   : sigma_3_par, 
    'perp1' : sigma_3_perp_1, 
    'perp2' : sigma_3_perp_2
  }

  # ================================================================
  # sigma_4: binned correlation uncertainty
  # sigma'''' = (1/n) * sqrt(sum_j sigma'''^2) per velocity bin
  # Shape : (N_t, N_energy, N_theta, N_phi)
  # ================================================================  

  vpar_flat    = vpar_n.ravel()
  vperp_1_flat = np.nanmean(vperp_1, axis=0).ravel() / vth
  vperp_2_flat = np.nanmean(vperp_2, axis=0).ravel() / vth

  vel_mask = np.isfinite(vpar_flat) & np.isfinite(vperp_1_flat) & np.isfinite(vperp_2_flat)

  vpar_flat = vpar_flat[vel_mask]
  vperp_1_flat = vperp_1_flat[vel_mask]
  vperp_2_flat = vperp_2_flat[vel_mask]

  sample = np.stack([vpar_flat, vperp_1_flat, vperp_2_flat], axis=1)

  sumS2 = {k: np.zeros((nbins_par, nbins_perp1, nbins_perp2)) for k in ('par', 'perp1', 'perp2')}
  counts = np.histogramdd(sample, bins=[vpar_edges, vperp_1_edges, vperp_2_edges])

  for t_idx in range(n_t):
      for key in ('par', 'perp1', 'perp2'):
        s_t    = sigma_3[key].reshape(n_t, -1)[t_idx][vel_mask]
        finite = np.isfinite(s_t)
        if not np.any(finite):
          continue
        s_t_masked = np.where(finite, s_t, 0.0)
        s2, _ = np.histogramdd(
          sample, bins=[vpar_edges, vperp_1_edges, vperp_2_edges],
          weights=s_t_masked**2
        )
        sumS2[key] += s2
  
  mask = counts > counts_to_mask
  sigma_4 = {}
  for key in ('par', 'perp1', 'perp2'):
        s4 = np.full((nbins_par, nbins_perp1, nbins_perp2), np.nan)
        s4[mask] = (1 / counts[mask]) * np.sqrt(sumS2[key][mask])
        sigma_4[key] = s4

  # ================================================================
  # sigma_5: reduced correlation uncertainty
  # sigma''''' = dv * sqrt(sum_j,k sigma''''^2) 
  # Shape : (n_bins,)
  # ================================================================  

  sigma_5_par   = dvperp1 * dvperp2 * np.sqrt(np.nansum(sigma_4['par']**2, axis=(1,2)))
  sigma_5_perp1 = dvpar   * dvperp2 * np.sqrt(np.nansum(sigma_4['perp1']**2, axis=(0, 2)))
  sigma_5_perp2 = dvpar   * dvperp1 * np.sqrt(np.nansum(sigma_4['perp2']**2, axis=(0, 1)))

  sigma_5 = {
         'par'   : sigma_5_par,
         'perp1' : sigma_5_perp1,
         'perp2' : sigma_5_perp2
         }

  # ================================================================
  # sigma_6: phase space transform uncertainty
  # sigma'''''' = (|v| / (2 * dv)) * sigma'''''
  # Shape : (n_bins,)
  # ================================================================ 

  sigma_6_par   = (np.abs(vpar_phys) / (2 * dvpar)) * sigma_5['par']
  sigma_6_perp1 = (np.abs(vperp_1_phys) / (2 * dvperp1)) * sigma_5['perp1']
  sigma_6_perp2 = (np.abs(vperp_2_phys) / (2 * dvperp2)) * sigma_5['perp2']

  sigma_6 = {
         'par'   : sigma_6_par,
         'perp1' : sigma_6_perp1,
         'perp2' : sigma_6_perp2
         }

  # ================================================================
  # sigma_7: time averaged correlation uncertainty
  # sigma''''''' = (1/n) * sigma'''''', n = tau / 0.03
  # Shape : (n_bins,)
  # ================================================================ 

  tau = t_dist[-1] - t_dist[0]
  n_d = max(int(tau / 0.03), 1)

  sigma_7_par   = (1 / n_d) * sigma_6['par']
  sigma_7_perp1 = (1 / n_d) * sigma_6['perp1']
  sigma_7_perp2 = (1 / n_d) * sigma_6['perp2']       

  sigma_7 = {
         'par'   : sigma_7_par,
         'perp1' : sigma_7_perp1,
         'perp2' : sigma_7_perp2
         }

  # ================================================================
  # sigma_f: final energy density transfer rate uncertainty
  # sigma_f = dv * sqrt(sum_j sigma'''''''^2)
  # Sum over v from -3 to +3 vth
  # ================================================================       

  within_3vth_vpar = np.abs(vpar_centres <= 3.0)
  sigma_f_par = dvpar * np.sqrt(np.nansum(sigma_7['par'][within_3vth_vpar]**2))

  within_3vth_vperp1 = np.abs(vperp_1_centres <= 3.0)
  sigma_f_perp1 = dvperp1 * np.sqrt(np.nansum(sigma_7['perp1'][within_3vth_vperp1]**2))

  within_3vth_vperp2 = np.abs(vperp_2_centres <= 3.0)
  sigma_f_perp2 = dvperp2 * np.sqrt(np.nansum(sigma_7['perp2'][within_3vth_vperp2]**2))

  sigma_f = {
         'par'   : sigma_f_par,
         'perp1' : sigma_f_perp1,
         'perp2' : sigma_f_perp2
         }
          

  # ================================================================
  # BUILDING RETURN DICTIONARY
  # ================================================================  
  
  uncertainty = {
          'sigma3' : sigma_3,
          'sigma4' : sigma_4,
          'sigma5' : sigma_5,
          'sigma6' : sigma_6,
          'sigma7' : sigma_7,
          'sigmaf' : sigma_f
         }

  return uncertainty
