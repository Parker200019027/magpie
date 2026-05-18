def fpc_uncertainty(dist,
                    eigen,
                    vth,
                    ve0,
                    edges,
                    spacecraft_id=1,
                    species='electron',
                    projection='gyro',
                    counts_to_mask=0,
                    ecut=None,
                    result=None,
                    direction='parallel'):
    '''
    Propagates Poisson counting uncertainties through the FPC pipeline
    following Afshari et al. Table B1 and Appendix B
    https://doi.org/10.1029/2021JA029578.

    *Temporary Correct Version*
    
    The electric field is treated as exact (no systematic uncertainty),
    consistent with Afshari et al. Appendix B. All uncertainty propagates
    from the Poisson counting statistics reported in the FPI level 2
    disterr variable.

    Two projections are supported, matching field_particle_correlation:

    - 'gyro'      : 2D (v_par, |v_perp|) grid. sigma_5 follows Afshari
                    et al. Table B1 exactly, with cylindrical 2pi*v_perp
                    weighting. Returns sigma_perp as the uncertainty on
                    the reduced perpendicular line C(v_perp).
    - 'cartesian' : 3D (v_par, v_perp1, v_perp2) grid. sigma_5 is the
                    cartesian analogue, integrating sigma_4 in quadrature
                    over the appropriate axes for each component.

    The signal_threshold parameter controls which bins are included when
    integrating sigma_4 into sigma_5 and sigma_perp. Velocity-space bins
    at the edges of the distribution are often sparsely sampled — only
    a handful of particles land there by chance. Because sigma_4 is
    proportional to 1/counts, these bins have large individual
    uncertainties but contribute negligible signal. When sigma_4 is
    integrated in quadrature across velocity space, these outlier bins
    can dominate the sum and produce an unrealistically large integrated
    uncertainty that bears no relation to the actual signal quality.

    The threshold works by masking any sigma_4 bin where the corresponding
    c_binned value is less than signal_threshold * max(|c_binned|). Only
    bins that contribute meaningfully to the signal are then included in
    the quadrature sum. The raw sigma_4 array is always returned unmasked
    so per-bin information is preserved. Set signal_threshold=0 to disable.

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
        computation.
    ve0 : array-like
        Mean bulk velocity in m/s, shape (3,).
    edges : dict
        Velocity grid edges from field_particle_correlation. Must contain
        'vpar'. Gyro mode also requires 'vperp'. Cartesian mode also
        requires 'vperp1' and 'vperp2'.
    spacecraft_id : int, optional
        MMS spacecraft ID (1-4). Default is 1.
    species : str, optional
        'ion' or 'electron'. Default is 'electron'.
    projection : str, optional
        'gyro' or 'cartesian'. Must match the projection used in
        field_particle_correlation. Default is 'gyro'.
    counts_to_mask : int, optional
        Minimum bin count threshold below which sigma_4 is set to NaN.
        Default is 0.
    signal_threshold : float, optional
        Fractional threshold relative to the peak signal. Bins in sigma_4
        where the corresponding c_binned value is below
        signal_threshold * max(|c_binned|) are excluded from the
        quadrature integration into sigma_5 and sigma_perp. Default 1e-3.
        Set to 0 to disable.
    ecut : float or None, optional
        Optional photoelectron energy cut in eV. Default is None.

    Returns
    -------
    uncertainty : dict
        Keys:
            'sigma3' : dict with keys 'par', 'perp1', 'perp2'
                       Shape (N_t, N_energy, N_theta, N_phi).
            'sigma4' : dict with keys 'par', 'perp1', 'perp2'
                       Gyro:      shape (nbins_par, nbins_perp)
                       Cartesian: shape (nbins_par, nbins_perp1, nbins_perp2)
                       Raw binned uncertainty, unmasked.
            'sigma5' : dict with keys 'par', 'perp1', 'perp2'
                       Shape (nbins,). Computed after signal threshold masking.
            'sigma6' : dict, same shape as sigma5.
            'sigma7' : dict, same shape as sigma5.
            'sigmaf' : dict with keys 'par', 'perp1', 'perp2' — scalars.
            'sigma_perp' : dict with keys 'perp1', 'perp2' [gyro only]
                       Uncertainty on the reduced perpendicular line C(v_perp).
                       None in cartesian mode.

    Raises
    ------
    ValueError
        If required tplot variables cannot be retrieved, disterr is
        unavailable, the pre-processed E field has not been stored, or
        projection is invalid.

    Notes
    -----
    Requires field_particle_correlation to have been run first so that
    'e_field_transformed_smooth' exists in the tplot store.
    Ion cadence uses dt=0.15s for n_d; electron cadence uses dt=0.03s.
    '''

    if projection not in ('gyro', 'cartesian'):
        raise ValueError(
            f"projection must be 'gyro' or 'cartesian', got '{projection}'"
        )

    if species == 'electron':
        q            = -1.60217663e-19
        mass         =  9.10938371e-31
        disterr_tvar = f'mms{spacecraft_id}_des_disterr_brst'
        dt_dist      = 0.03
    else:
        q            =  1.60217663e-19
        mass         =  1.67262192e-27
        disterr_tvar = f'mms{spacecraft_id}_dis_disterr_brst'
        dt_dist      = 0.15

    # =========================================================================
    # VELOCITY GRID
    # =========================================================================

    vpar_edges   = edges['vpar']
    nbins_par    = len(vpar_edges) - 1
    vpar_centres = 0.5 * (vpar_edges[:-1] + vpar_edges[1:])
    dvpar        = (vpar_edges[1] - vpar_edges[0])
    vpar_phys    = vpar_centres * vth

    if projection == 'gyro':
        vperp_edges   = edges['vperp']
        nbins_perp    = len(vperp_edges) - 1
        vperp_centres = 0.5 * (vperp_edges[:-1] + vperp_edges[1:])
        dvperp        = (vperp_edges[1] - vperp_edges[0])
        vperp_phys    = vperp_centres * vth
    else:
        vperp1_edges   = edges['vperp1']
        vperp2_edges   = edges['vperp2']
        nbins_perp1    = len(vperp1_edges) - 1
        nbins_perp2    = len(vperp2_edges) - 1
        vperp1_centres = 0.5 * (vperp1_edges[:-1] + vperp1_edges[1:])
        vperp2_centres = 0.5 * (vperp2_edges[:-1] + vperp2_edges[1:])
        dvperp1        = (vperp1_edges[1] - vperp1_edges[0])
        dvperp2        = (vperp2_edges[1] - vperp2_edges[0])
        vperp1_phys    = vperp1_centres * vth
        vperp2_phys    = vperp2_centres * vth

    # =========================================================================
    # LOAD DISTERR AND E FIELD
    # =========================================================================

    disterr_data = get_data(disterr_tvar)
    if disterr_data is None:
        raise ValueError(
            f"Could not retrieve '{disterr_tvar}'. "
            "Ensure FPI burst data has been loaded."
        )

    e_data = get_data('e_field_transformed_smooth')
    if e_data is None:
        raise ValueError(
            "Could not retrieve 'e_field_transformed_smooth'. "
            "Ensure field_particle_correlation has been run first."
        )

    # =========================================================================
    # DISTRIBUTION TIMES AND VELOCITY PROJECTION
    # =========================================================================

    t_dist = np.array(
        [(d['start_time'] + d['end_time']) * 0.5 for d in dist], dtype=float
    )
    n_t = len(t_dist)

    energy = np.array([d['energy'] for d in dist], dtype=float)
    theta  = np.deg2rad(np.array([dist[0]['theta']], dtype=float))
    phi    = np.deg2rad(np.array([dist[0]['phi']],   dtype=float))

    if ecut is not None:
        energy = np.where(energy > ecut, energy, np.nan)
    energy = np.where(energy > 0, energy, 1e-12)

    v    = np.sqrt(2 * energy * np.abs(q) / mass)
    vx   = v * np.cos(theta[0]) * np.cos(phi[0]) - ve0[0]
    vy   = v * np.cos(theta[0]) * np.sin(phi[0]) - ve0[1]
    vz   = v * np.sin(theta[0]) - ve0[2]
    vvec = np.stack([vx, vy, vz], axis=-1)

    vpar    = np.tensordot(vvec, eigen[0], axes=([-1], [0]))
    vperp_1 = np.tensordot(vvec, eigen[1], axes=([-1], [0]))
    vperp_2 = np.tensordot(vvec, eigen[2], axes=([-1], [0]))
    vperp   = np.sqrt(vperp_1**2 + vperp_2**2)

    vpar_mean    = np.nanmean(vpar,    axis=0)
    vperp_1_mean = np.nanmean(vperp_1, axis=0)
    vperp_2_mean = np.nanmean(vperp_2, axis=0)

    vpar_n   = vpar[0].ravel()   / vth
    vperp_n  = vperp[0].ravel() / vth
    vperp1_n = np.nanmean(vperp_1, axis=0).ravel() / vth
    vperp2_n = np.nanmean(vperp_2, axis=0).ravel() / vth

    # =========================================================================
    # INTERPOLATE DISTERR AND E FIELD
    # =========================================================================

    store_data('_disterr_full', data={'x': disterr_data.times, 'y': disterr_data.y})
    tinterpol('_disterr_full', t_dist, newname='_disterr_interp')
    disterr_interp = get_data('_disterr_interp')
    if disterr_interp is None:
        raise ValueError("Could not interpolate disterr onto distribution times.")

    sample_rate = 1 / np.median(np.diff(get_data(f'mms{spacecraft_id}_d{species[0]}s_bulkv_gse_brst').times))

    sos = scipy.signal.butter(5, 1, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, e_data.y, axis=0)
    
    store_data('_e_unc_full', data={'x': e_data.times, 'y': filtered_data})
    tinterpol('_e_unc_full', t_dist, newname='_e_unc_interp')
    e_interp = get_data('_e_unc_interp')
    if e_interp is None:
        raise ValueError("Could not interpolate E field onto distribution times.")

    epar    = np.dot(e_interp.y * 1e-3, eigen[0])
    eperp_1 = np.dot(e_interp.y * 1e-3, eigen[1])
    eperp_2 = np.dot(e_interp.y * 1e-3, eigen[2])

    # =========================================================================
    # STEP 1 — sigma: disterr in SI units
    # Shape: (N_t, N_energy, N_theta, N_phi)
    # =========================================================================

    sigma = disterr_interp.y * 1e12
    sigma = sigma.transpose(0, 1, 3, 2)

    f_data = np.array([d['data'] * 1e12 for d in dist], dtype=float)
    sigma_safe = np.where(sigma > 0, sigma, np.nan)
    counts_raw = (f_data / sigma_safe) ** 2
    sigma = np.where(counts_raw >= counts_to_mask, sigma, np.nan)

    sigma1 = 1/n_t * np.sqrt(np.nansum(sigma**2, axis=0))

    sigma2 = np.sqrt(sigma1**2 + sigma**2)

    sigma3_par = q * vpar * sigma2 * epar[:, np.newaxis, np.newaxis, np.newaxis]
    sigma3_perp1 = q * vperp_1 * sigma2 * eperp_1[:, np.newaxis, np.newaxis, np.newaxis]
    sigma3_perp2 = q * vperp_2 * sigma2 * eperp_2[:, np.newaxis, np.newaxis, np.newaxis]

    if direction == 'parallel':

        sigma3 = sigma3_par

        counts, _, _ = np.histogram2d(
            vpar_n, vperp_n, bins=[vpar_edges, vperp_edges]
        )
    
        sigma_n = np.nanmean(sigma3, axis=0).ravel()
        
        sumS, _, _ = np.histogram2d(
            vpar_n, vperp_n, bins=[vpar_edges, vperp_edges], weights=sigma_n
        )
    
        ''' 
        sigma4 = np.full_like(sumS, np.nan, dtype=float)
        
        mask = counts > counts_to_mask
    
        sigma4[mask] = np.sqrt(np.nansum(sumS[mask]**2)) / counts[mask]
        '''
    
        sigma4 = np.full((nbins_par, nbins_perp), np.nan)
        sumS2 = np.zeros((nbins_par, nbins_perp))
        counts = np.zeros((nbins_par, nbins_perp))
    
        sigma6_sum2 = np.zeros(nbins_par)
        sigma4_sum2 = np.zeros((nbins_par, nbins_perp))
        counts_total = np.zeros((nbins_par, nbins_perp))
        
        for i in range(n_t):
            sigma4_i = np.full((nbins_par, nbins_perp), np.nan)
            
            sigma_n_i = sigma3[i].ravel()
            s, _, _ = np.histogram2d(vpar_n, vperp_n, bins=[vpar_edges, vperp_edges], weights=sigma_n_i)
            c, _, _ = np.histogram2d(vpar_n, vperp_n, bins=[vpar_edges, vperp_edges])
    
            if i==0:
                print(c.max())
            
            sigma4_i = s / c
    
            sigma5_i = 2 * np.pi * dvperp * vth * np.sqrt(np.nansum((vperp_centres * vth * sigma4_i)**2, axis=1))
    
            sigma6_i = np.abs(vpar_centres / (2 * dvpar)) * sigma5_i
            
            sigma6_sum2 += np.where(np.isnan(sigma6_i), 0, sigma6_i**2)
            sigma4_sum2 += np.where(np.isnan(sigma4_i), 0, sigma4_i**2)
            counts_total += c
        
        sigma7 = (1 / n_t) * np.sqrt(sigma6_sum2)
        
        # Time-averaged sigma4 and sigma5 for inspection
        sigma4 = np.full((nbins_par, nbins_perp), np.nan)
        sigma4 = (1 / (n_t * 0.03)) * np.sqrt(sigma4_sum2)
        
        sigma5 = 2 * np.pi * dvperp * vth * np.sqrt(np.nansum((vperp_centres * vth * sigma4)**2, axis=1))
        sigma6 = np.abs(vpar_centres / (2 * dvpar)) * sigma5
    
        return sigma7

    elif direction == 'perpendicular':

        sigma4_sum2_perp1  = np.zeros((nbins_par, nbins_perp))
        sigma4_sum2_perp2  = np.zeros((nbins_par, nbins_perp))
        sigma6_sum2        = np.zeros(nbins_perp)
        counts_total       = np.zeros((nbins_par, nbins_perp))
    
        for i in range(n_t):
            s1, _, _ = np.histogram2d(vpar_n, vperp_n, bins=[vpar_edges, vperp_edges],
                                       weights=sigma3_perp1[i].ravel())
            s2, _, _ = np.histogram2d(vpar_n, vperp_n, bins=[vpar_edges, vperp_edges],
                                       weights=sigma3_perp2[i].ravel())
            c,  _, _ = np.histogram2d(vpar_n, vperp_n, bins=[vpar_edges, vperp_edges])
    
            mask_i    = c > 0
            sigma4_i_perp1 = np.where(mask_i, s1 / c, np.nan)
            sigma4_i_perp2 = np.where(mask_i, s2 / c, np.nan)
    
            # sigma5: integrate each component over vpar — shape (nbins_perp,)
            sigma5_i_perp1 = dvpar * vth * np.sqrt(np.nansum(sigma4_i_perp1**2, axis=0))
            sigma5_i_perp2 = dvpar * vth * np.sqrt(np.nansum(sigma4_i_perp2**2, axis=0))
    
            # sigma6: apply Eq. 6 to each component then sum in quadrature
            dc_dvperp_1  = np.gradient(sigma5_i_perp1, vperp_phys)
            dc_dvperp_2  = np.gradient(sigma5_i_perp2, vperp_phys)
            sigma6_i_perp1 = np.abs(-0.5 * vperp_phys * dc_dvperp_1 + sigma5_i_perp1 / 2)
            sigma6_i_perp2 = np.abs(-0.5 * vperp_phys * dc_dvperp_2 + sigma5_i_perp2 / 2)
    
            sigma6_i = np.sqrt(sigma6_i_perp1**2 + sigma6_i_perp2**2)
    
            sigma6_sum2  += np.where(np.isnan(sigma6_i), 0, sigma6_i**2)
            sigma4_sum2_perp1 += np.where(np.isnan(sigma4_i_perp1), 0, sigma4_i_perp1**2)
            sigma4_sum2_perp2 += np.where(np.isnan(sigma4_i_perp2), 0, sigma4_i_perp2**2)
            counts_total += c
    
        sigma7 = (1 / n_t) * np.sqrt(sigma6_sum2)  # shape (nbins_perp,)
    
        return sigma7
