import numpy as np
from pyspedas import get_data, store_data, tinterpol, subtract
import scipy.signal


def boxcar_averager(data_times, data, interp_times):
    '''
    Boxcar averages data onto interp_times using the nearest timestamps in
    data_times, with window width determined by interp_times / data_times.

    Parameters
    ----------
    data_times : array-like
        Timestamps corresponding to data points.
    data : array-like
        Data array to be averaged (first axis must match length of data_times).
    interp_times : array-like
        Target timestamps for interpolated/averaged output.

    Returns
    -------
    interp_times : ndarray
        Input interpolation times (returned for convenience).
    averaged : ndarray
        Boxcar averaged data at interp_times.

    Raises
    ------
    TypeError
        If inputs cannot be converted to numpy arrays.
    ValueError
        If inputs are invalid (empty, mismatched dimensions, non-monotonic).
    RuntimeError
        If an empty averaging window is encountered.
    '''

    try:
        data_times   = np.asarray(data_times)
        data         = np.asarray(data)
        interp_times = np.asarray(interp_times)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {e}")

    if data_times.ndim != 1:
        raise ValueError(f"data_times must be 1D, got {data_times.ndim}D")
    if interp_times.ndim != 1:
        raise ValueError(f"interp_times must be 1D, got {interp_times.ndim}D")
    if data.shape[0] != len(data_times):
        raise ValueError(
            f"First dimension of data ({data.shape[0]}) must match "
            f"length of data_times ({len(data_times)})"
        )
    if len(data_times) < 2:
        raise ValueError("data_times must have at least 2 points")
    if len(interp_times) < 2:
        raise ValueError("interp_times must have at least 2 points")
    if not np.all(np.diff(data_times) > 0):
        raise ValueError("data_times must be strictly monotonically increasing")
    if not np.all(np.diff(interp_times) > 0):
        raise ValueError("interp_times must be strictly monotonically increasing")

    dt1   = np.median(np.diff(data_times))
    dt2   = np.median(np.diff(interp_times))
    width = int(round(dt2 / dt1))

    if width < 1:
        raise ValueError(
            f"Computed window width ({width}) is less than 1. "
            f"interp_times spacing (dt2={dt2:.3e}) may be too small "
            f"compared to data_times spacing (dt1={dt1:.3e})"
        )

    half     = width // 2
    idx1     = np.clip(np.searchsorted(data_times, interp_times), 0, len(data_times) - 1)
    averaged = np.zeros((len(interp_times),) + data.shape[1:])

    for i, centre_idx in enumerate(idx1):
        start = max(centre_idx - half, 0)
        end   = min(centre_idx + half + 1, len(data))
        if end <= start:
            raise RuntimeError(
                f"Empty averaging window at index {i}: start={start}, end={end}"
            )
        averaged[i] = np.mean(data[start:end], axis=0)

    return interp_times, averaged


def lorentz(e_field, b_field, bulkv, spintone=None):
    '''
    Transforms the electric field from the spacecraft frame to the plasma frame
    using the Lorentz transformation: E_plasma = E_sc + (v x B).

    The E field is first smoothed (boxcar averaged) down to the B field cadence
    before the rotation, so that E and B share a common time axis at the FGM
    resolution. No interpolation of B is required.

    Parameters
    ----------
    e_field : str
        Tplot variable name for the electric field in the spacecraft frame (mV/m).
    b_field : str
        Tplot variable name for the magnetic field in the spacecraft frame (nT).
    bulkv : str
        Tplot variable name for the ion bulk velocity (km/s).
    spintone : str, optional
        Tplot variable name for the spintone product (km/s). Default is None.

    Returns
    -------
    smooth_name : str
        Tplot variable name of the smoothed Lorentz-transformed E field.
    b_xyz : ndarray
        Native-cadence magnetic field time series, shape (N, 3), in nT.
    b_times : ndarray
        Timestamps corresponding to b_xyz.
    v_ms_avg : ndarray
        Mean bulk velocity vector, shape (3,), in m/s.

    Raises
    ------
    TypeError
        If any tplot variable name argument is not a string.
    ValueError
        If any tplot variable cannot be retrieved, has wrong shape, or contains
        non-finite values.

    Notes
    -----
    b_times is returned alongside b_xyz so that the caller can interpolate
    B onto arbitrary timestamps (required for instantaneous FAC mode).
    '''

    for arg_name, arg_val in [('e_field', e_field), ('b_field', b_field), ('bulkv', bulkv)]:
        if not isinstance(arg_val, str):
            raise TypeError(
                f"'{arg_name}' must be a string tplot variable name, "
                f"got {type(arg_val).__name__}"
            )
    if spintone is not None and not isinstance(spintone, str):
        raise TypeError(f"'spintone' must be a string or None, got {type(spintone).__name__}")

    e_sc = get_data(e_field)
    if e_sc is None:
        raise ValueError(f"Could not retrieve tplot variable '{e_field}'.")
    b_sc = get_data(b_field)
    if b_sc is None:
        raise ValueError(f"Could not retrieve tplot variable '{b_field}'.")
    bulkv_data = get_data(bulkv)
    if bulkv_data is None:
        raise ValueError(f"Could not retrieve tplot variable '{bulkv}'.")

    if e_sc.y.ndim != 2 or e_sc.y.shape[1] < 3:
        raise ValueError(f"'{e_field}' must be 2D with >=3 components, got shape {e_sc.y.shape}")
    if b_sc.y.ndim != 2 or b_sc.y.shape[1] < 3:
        raise ValueError(f"'{b_field}' must be 2D with >=3 components, got shape {b_sc.y.shape}")

    if not np.all(np.isfinite(e_sc.y)):
        raise ValueError(f"'{e_field}' contains NaN or Inf values.")

    # --- Spintone correction (operates on original bulkv cadence) ---
    if spintone is not None:
        spintone_data = get_data(spintone)
        if spintone_data is None:
            raise ValueError(f"Could not retrieve spintone variable '{spintone}'.")
        subtract(bulkv, spintone, newname='bulkv_corrected')
        v_corrected_data = get_data('bulkv_corrected')
        if v_corrected_data is None:
            raise ValueError("Spintone subtraction failed.")
        v_ms_avg = np.mean(v_corrected_data.y, axis=0) * 1e3
    else:
        v_ms_avg = np.mean(bulkv_data.y, axis=0) * 1e3

    if v_ms_avg.shape != (3,):
        raise ValueError(f"Averaged bulk velocity must have 3 components, got {v_ms_avg.shape}")
    if not np.all(np.isfinite(v_ms_avg)):
        raise ValueError("Averaged bulk velocity contains NaN or Inf values.")

    # --- Smooth E field down to B cadence (no B interpolation needed) ---
    smooth_times, smooth_y = boxcar_averager(e_sc.times, e_sc.y, b_sc.times)
    smooth_name = 'e_field_smooth'
    store_data(smooth_name, data={'x': b_sc.times, 'y': smooth_y})
    e_smooth = get_data(smooth_name)
    if e_smooth is None:
        raise ValueError("Failed to store 'e_field_smooth'.")

    # --- Use B at its native cadence — no tinterpol required ---
    b_xyz   = b_sc.y[:, 0:3]
    b_times = b_sc.times
    if not np.all(np.isfinite(b_xyz)):
        raise ValueError("B field contains NaN or Inf values.")

    # --- Lorentz rotation: E and B now share the B cadence time axis ---
    e_lorentz     = (e_smooth.y * 1e-3) + np.cross(v_ms_avg, b_xyz * 1e-9)
    e_lorentz_mvm = e_lorentz * 1e3

    store_data('e_field_transformed', data={'x': b_sc.times, 'y': e_lorentz_mvm})
    transformed_data = get_data('e_field_transformed')
    if transformed_data is None:
        raise ValueError("Failed to store 'e_field_transformed'.")

    # --- Smooth transformed E down to DIS/DES cadence ---
    smooth_times, smooth_y = boxcar_averager(
        transformed_data.times, transformed_data.y, bulkv_data.times
    )
    smooth_name = 'e_field_transformed_smooth'
    store_data(smooth_name, data={'x': bulkv_data.times, 'y': smooth_y})
    if get_data(smooth_name) is None:
        raise ValueError(f"Failed to store '{smooth_name}'.")

    return smooth_name, b_xyz, b_times, v_ms_avg


def eigenvectors(b_xyz, v_ms_avg):
    '''
    Computes orthonormal FAC basis vectors from mean B and bulk velocity.

    e_par  : parallel to mean B
    e_perp2: parallel to (B x v)
    e_perp1: completes the right-handed set

    Parameters
    ----------
    b_xyz : array-like
        Magnetic field time series, shape (N, 3).
    v_ms_avg : array-like
        Mean ion bulk velocity vector, shape (3,).

    Returns
    -------
    eigen : list of ndarray
        [e_par, e_perp1, e_perp2], each a unit vector of shape (3,).

    Raises
    ------
    TypeError
        If inputs cannot be converted to numpy arrays.
    ValueError
        If inputs have wrong shape, contain non-finite values, or produce
        degenerate cross products.
    '''

    try:
        b_xyz    = np.asarray(b_xyz, dtype=float)
        v_ms_avg = np.asarray(v_ms_avg, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {e}")

    if b_xyz.ndim != 2 or b_xyz.shape[1] != 3:
        raise ValueError(f"b_xyz must have shape (N, 3), got {b_xyz.shape}")
    if v_ms_avg.shape != (3,):
        raise ValueError(f"v_ms_avg must have shape (3,), got {v_ms_avg.shape}")
    if not np.all(np.isfinite(b_xyz)):
        raise ValueError("b_xyz contains NaN or Inf values.")
    if not np.all(np.isfinite(v_ms_avg)):
        raise ValueError("v_ms_avg contains NaN or Inf values.")

    b0      = np.nanmean(b_xyz, axis=0)
    b0_norm = np.linalg.norm(b0)
    if b0_norm < 1e-10:
        raise ValueError(f"Mean B magnitude is effectively zero ({b0_norm:.2e}).")
    e_par = b0 / b0_norm

    v_norm = np.linalg.norm(v_ms_avg)
    if v_norm < 1e-10:
        raise ValueError(f"v_ms_avg magnitude is effectively zero ({v_norm:.2e}).")
    cos_angle = np.dot(b0, v_ms_avg) / (b0_norm * v_norm)
    if abs(cos_angle) > 1 - 1e-6:
        raise ValueError(
            "v_ms_avg is parallel to mean B — cannot define perpendicular directions."
        )

    bxv     = np.cross(b0, v_ms_avg)
    e_perp2 = bxv / np.linalg.norm(bxv)
    bxvxb   = np.cross(bxv, b0)
    e_perp1 = bxvxb / np.linalg.norm(bxvxb)

    return [e_par, e_perp1, e_perp2]


def _instantaneous_basis(b_vec, v_ms_avg):
    '''
    Computes an orthonormal FAC basis from a single instantaneous B vector.

    Uses the same convention as eigenvectors():
        e_par   : parallel to B
        e_perp2 : parallel to (B x v)
        e_perp1 : completes the right-handed set (= (B x v) x B, normalised)

    v_ms_avg is used only as a reference direction to define the perpendicular
    plane; it is not subtracted from any velocity grid.

    Parameters
    ----------
    b_vec : array-like, shape (3,)
        Instantaneous magnetic field vector (any consistent units).
    v_ms_avg : array-like, shape (3,)
        Mean bulk velocity vector in m/s.  Used as perpendicular reference only.

    Returns
    -------
    e_par : ndarray, shape (3,)
    e_perp1 : ndarray, shape (3,)
    e_perp2 : ndarray, shape (3,)

    Raises
    ------
    ValueError
        If b_vec magnitude is effectively zero, or if b_vec is parallel to
        v_ms_avg (degenerate perpendicular plane).
    '''

    b_vec    = np.asarray(b_vec,    dtype=float)
    v_ms_avg = np.asarray(v_ms_avg, dtype=float)

    b_norm = np.linalg.norm(b_vec)
    if b_norm < 1e-10:
        raise ValueError(f"Instantaneous |B| is effectively zero ({b_norm:.2e}).")
    e_par = b_vec / b_norm

    v_norm    = np.linalg.norm(v_ms_avg)
    cos_angle = np.dot(e_par, v_ms_avg) / (v_norm + 1e-30)
    if abs(cos_angle) > 1 - 1e-6:
        raise ValueError(
            "Instantaneous B is parallel to v_ms_avg — degenerate perpendicular plane."
        )

    bxv     = np.cross(b_vec, v_ms_avg)
    e_perp2 = bxv / np.linalg.norm(bxv)
    bxvxb   = np.cross(bxv, b_vec)
    e_perp1 = bxvxb / np.linalg.norm(bxvxb)

    return e_par, e_perp1, e_perp2


def highpass(data, cutoff, sample_rate, order):
    '''
    Applies a zero-phase Butterworth highpass filter to data.

    Parameters
    ----------
    data : array-like
        Input data. Filtering applied along axis=0.
    cutoff : float
        Cutoff frequency in Hz.
    sample_rate : float
        Sampling frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    filtered_data : ndarray
        Highpass filtered data, same shape as input.

    Raises
    ------
    TypeError
        If inputs cannot be converted to expected types.
    ValueError
        If filter parameters are invalid or data is too short.
    '''

    try:
        data = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert data to numpy array: {e}")

    try:
        cutoff      = float(cutoff)
        sample_rate = float(sample_rate)
    except (ValueError, TypeError) as e:
        raise TypeError(f"cutoff and sample_rate must be numeric: {e}")

    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError(f"order must be a positive integer, got {order}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if cutoff >= sample_rate / 2:
        raise ValueError(
            f"cutoff ({cutoff} Hz) must be less than Nyquist ({sample_rate / 2} Hz)."
        )
    if data.size == 0:
        raise ValueError("data cannot be empty.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data contains NaN or Inf values.")

    min_samples = 3 * (2 * order) + 1
    if data.shape[0] < min_samples:
        raise ValueError(
            f"data has too few samples ({data.shape[0]}) for order {order}. "
            f"Minimum: {min_samples}."
        )

    sos           = scipy.signal.butter(order, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data, axis=0)
    return filtered_data


SPECIES = {
    'ion': {
        'q':  1.60217663e-19,
        'm':  1.67262192e-27,
        'label': 'ion'
    },
    'electron': {
        'q': -1.60217663e-19,
        'm':  9.10938371e-31,
        'label': 'electron'
    }
}


def _compute_vth(species, spacecraft_id, direction):
    '''
    Compute thermal velocity from mean temperature.

    Parameters
    ----------
    species : str
        'ion' or 'electron'.
    spacecraft_id : int
        MMS spacecraft ID (1-4).
    direction : str
        'parallel' uses temppara; 'perpendicular' uses tempperp.

    Returns
    -------
    vth : float
        Thermal velocity in m/s.

    Raises
    ------
    ValueError
        If the temperature tplot variable cannot be retrieved.
    '''

    q         = SPECIES[species]['q']
    mass      = SPECIES[species]['m']
    inst      = 'dis' if species == 'ion' else 'des'
    temp_dir  = 'para' if direction == 'parallel' else 'perp'
    temp_tvar = f'mms{spacecraft_id}_{inst}_temp{temp_dir}_brst'
    temp_data = get_data(temp_tvar)
    if temp_data is None:
        raise ValueError(f"Could not retrieve temperature variable '{temp_tvar}'.")
    return np.sqrt(2 * np.abs(q) * np.mean(temp_data.y) / mass)


def _project_gyro(c_dict, vpar, vperp, f,
                  vth, vpar_edges, vperp_edges):
    '''
    Projects a dict of correlation arrays and the distribution onto a 2D
    (v_par, |v_perp|) grid, assuming gyrotropy.

    The distribution histogram is computed once and shared across all
    correlation components.

    Parameters
    ----------
    c_dict : dict
        Dict of time-averaged correlation arrays, each shape
        (N_energy, N_theta, N_phi). Keys: 'par', 'perp1', 'perp2'.
    vpar : ndarray
        Parallel velocity, shape (N_time, N_energy, N_theta, N_phi) or
        (N_energy, N_theta, N_phi), in m/s.
    vperp : ndarray
        Perpendicular speed |v_perp|, same shape as vpar, in m/s.
    f : ndarray
        Distribution function, shape (N_time, N_energy, N_theta, N_phi).
    vth : float
        Thermal velocity in m/s.
    vpar_edges : ndarray
        Parallel velocity bin edges, normalised by vth, shape (nbins+1,).
    vperp_edges : ndarray
        Perpendicular speed bin edges, normalised by vth, shape (nbins+1,).

    Returns
    -------
    result : dict
        Keys 'par', 'perp1', 'perp2', each a dict with keys:
            'c_binned' : ndarray, shape (nbins, nbins)
            'sumC'     : ndarray, shape (nbins, nbins)
        Also contains:
            'counts'   : ndarray, shape (nbins, nbins)
            'f_binned' : ndarray, shape (nbins, nbins)
            'sumF'     : ndarray, shape (nbins, nbins)
    c_dict : dict
        The input c_dict, returned unchanged for diagnostic access.
    '''

    f_flat     = np.nanmean(f,     axis=0).ravel()
    vpar_flat  = np.nanmean(vpar,  axis=0).ravel() / vth
    vperp_flat = np.nanmean(vperp, axis=0).ravel() / vth

    vel_mask = np.isfinite(vpar_flat) & np.isfinite(vperp_flat)

    vpar_flat  = vpar_flat[vel_mask]
    vperp_flat = vperp_flat[vel_mask]
    f_flat     = f_flat[vel_mask]

    counts, _, _ = np.histogram2d(
        vpar_flat, vperp_flat, bins=[vpar_edges, vperp_edges]
    )
    sumF, _, _ = np.histogram2d(
        vpar_flat, vperp_flat, bins=[vpar_edges, vperp_edges], weights=f_flat
    )
    f_binned = sumF / np.where(counts > 0, counts, np.nan)

    result = {'counts': counts, 'f_binned': f_binned, 'sumF': sumF}

    for key, c_arr in c_dict.items():
        c_flat = c_arr.ravel()[vel_mask]
        if not np.any(np.isfinite(c_flat)):
            raise ValueError(f"No finite correlation values for component '{key}'.")
        sumC, _, _ = np.histogram2d(
            vpar_flat, vperp_flat, bins=[vpar_edges, vperp_edges], weights=c_flat
        )
        c_binned = sumC / np.where(counts > 0, counts, np.nan)
        result[key] = {'c_binned': c_binned, 'sumC': sumC}

    return result, c_dict


def _project_cartesian(c_dict, vpar, vperp_1, vperp_2, f,
                        vth, vpar_edges, vperp1_edges, vperp2_edges):
    '''
    Projects a dict of correlation arrays and the distribution onto a 3D
    (v_par, v_perp1, v_perp2) Cartesian grid, making no gyrotropy assumption.

    The distribution histogram is computed once and shared across all
    correlation components.

    Parameters
    ----------
    c_dict : dict
        Dict of time-averaged correlation arrays, each shape
        (N_energy, N_theta, N_phi). Keys: 'par', 'perp1', 'perp2'.
    vpar : ndarray
        Parallel velocity, shape (N_time, N_energy, N_theta, N_phi) or
        (N_energy, N_theta, N_phi), in m/s.
    vperp_1 : ndarray
        First perpendicular component, same shape as vpar, in m/s.
    vperp_2 : ndarray
        Second perpendicular component, same shape as vpar, in m/s.
    f : ndarray
        Distribution function, shape (N_time, N_energy, N_theta, N_phi).
    vth : float
        Thermal velocity in m/s.
    vpar_edges : ndarray
        Parallel velocity bin edges, normalised by vth, shape (nbins+1,).
    vperp1_edges : ndarray
        v_perp1 bin edges, normalised by vth, shape (nbins+1,). Runs -vmax to vmax.
    vperp2_edges : ndarray
        v_perp2 bin edges, normalised by vth, shape (nbins+1,). Runs -vmax to vmax.

    Returns
    -------
    result : dict
        Keys 'par', 'perp1', 'perp2', each a dict with keys:
            'c_binned' : ndarray, shape (nbins, nbins, nbins)
            'sumC'     : ndarray, shape (nbins, nbins, nbins)
        Also contains:
            'counts'   : ndarray, shape (nbins, nbins, nbins)
            'f_binned' : ndarray, shape (nbins, nbins, nbins)
            'sumF'     : ndarray, shape (nbins, nbins, nbins)
    c_dict : dict
        The input c_dict, returned unchanged for diagnostic access.

    Notes
    -----
    Suggested post-processing projections (dvX = bin width in m/s):

    (v_perp1, v_perp2) plane — sum over v_par:
        dvpar = (vpar_edges[1] - vpar_edges[0]) * vth
        c_perp = np.nansum(c_binned * dvpar, axis=0)

    (v_par, v_perp1) plane — sum over v_perp2:
        dvperp2 = (vperp2_edges[1] - vperp2_edges[0]) * vth
        c_par_perp1 = np.nansum(c_binned * dvperp2, axis=2)

    (v_par, v_perp2) plane — sum over v_perp1:
        dvperp1 = (vperp1_edges[1] - vperp1_edges[0]) * vth
        c_par_perp2 = np.nansum(c_binned * dvperp1, axis=1)
    '''

    f_flat       = np.nanmean(f,       axis=0).ravel()
    vpar_flat    = np.nanmean(vpar,    axis=0).ravel() / vth
    vperp_1_flat = np.nanmean(vperp_1, axis=0).ravel() / vth
    vperp_2_flat = np.nanmean(vperp_2, axis=0).ravel() / vth

    vel_mask = (
        np.isfinite(vpar_flat)    &
        np.isfinite(vperp_1_flat) &
        np.isfinite(vperp_2_flat)
    )

    vpar_flat    = vpar_flat[vel_mask]
    vperp_1_flat = vperp_1_flat[vel_mask]
    vperp_2_flat = vperp_2_flat[vel_mask]
    f_flat       = f_flat[vel_mask]

    sample = np.stack([vpar_flat, vperp_1_flat, vperp_2_flat], axis=1)

    counts, _ = np.histogramdd(sample, bins=[vpar_edges, vperp1_edges, vperp2_edges])
    sumF,   _ = np.histogramdd(
        sample, bins=[vpar_edges, vperp1_edges, vperp2_edges], weights=f_flat
    )
    f_binned = sumF / np.where(counts > 0, counts, np.nan)

    result = {'counts': counts, 'f_binned': f_binned, 'sumF': sumF}

    for key, c_arr in c_dict.items():
        c_flat = c_arr.ravel()[vel_mask]
        if not np.any(np.isfinite(c_flat)):
            raise ValueError(f"No finite correlation values for component '{key}'.")
        sumC, _ = np.histogramdd(
            sample, bins=[vpar_edges, vperp1_edges, vperp2_edges], weights=c_flat
        )
        c_binned = sumC / np.where(counts > 0, counts, np.nan)
        result[key] = {'c_binned': c_binned, 'sumC': sumC}

    return result, c_dict


def _correlate_chunk(dist_chunk, e_filt_y, e_filt_times, eigen, vth, ve0,
                     species, counts_to_mask,
                     vpar_edges, vperp_edges, vperp1_edges, vperp2_edges,
                     nbins, projection, ecut, d3v,
                     fac_mode='fixed', b_xyz=None, b_times=None,
                     v_ms_avg=None):
    '''
    Core correlation computation for a single chunk of distributions, using
    pre-filtered E field data passed in directly. Always computes correlations
    for all three field components (par, perp1, perp2).

    Parameters
    ----------
    dist_chunk : list of dict
        Particle distribution dicts for this chunk.
    e_filt_y : ndarray
        Pre-filtered E field values, shape (N_e, 3), in mV/m.
    e_filt_times : ndarray
        Timestamps corresponding to e_filt_y.
    eigen : list of ndarray or None
        FAC basis vectors [e_par, e_perp1, e_perp2] for fac_mode='fixed'.
        Must be None when fac_mode='instantaneous'.
    vth : float
        Thermal velocity in m/s.
    ve0 : ndarray, shape (3,)
        Mean bulk velocity in m/s used for velocity frame shift.
    species : str
        'ion' or 'electron'.
    counts_to_mask : int
        Bin count threshold below which bins are NaN.
    vpar_edges : ndarray
        Parallel velocity bin edges normalised by vth.
    vperp_edges : ndarray
        Perpendicular speed bin edges normalised by vth (gyro mode).
    vperp1_edges : ndarray
        v_perp1 bin edges normalised by vth (cartesian mode).
    vperp2_edges : ndarray
        v_perp2 bin edges normalised by vth (cartesian mode).
    nbins : int
        Number of bins along each axis.
    projection : str
        'gyro' or 'cartesian'.
    ecut : float or None
        Low-energy cut in eV (photoelectron removal). None = no cut.
    d3v : ndarray
        Velocity-space volume element, shape (N_energy, N_phi, N_theta), in m^3/s^3.
        Used for direct integration of c_par_ts_raw.
    fac_mode : str, optional
        'fixed'         — use a single pre-computed FAC basis (eigen) for the
                          whole chunk (default, original behaviour).
        'instantaneous' — build a new FAC basis at every distribution timestep
                          from the interpolated B vector.
    b_xyz : ndarray or None
        Full-interval B field time series, shape (N_b, 3), in nT.
        Required when fac_mode='instantaneous'.
    b_times : ndarray or None
        Timestamps for b_xyz. Required when fac_mode='instantaneous'.
    v_ms_avg : ndarray or None
        Mean bulk velocity (3,) in m/s, used as the perpendicular reference
        when constructing each instantaneous basis.
        Required when fac_mode='instantaneous'.

    Returns
    -------
    result : dict
        Output from _project_gyro or _project_cartesian. Contains keys
        'par', 'perp1', 'perp2' (each with 'c_binned' and 'sumC'), plus
        'counts', 'f_binned', 'sumF'.
    c_dict : dict
        Pre-binning correlation arrays keyed by 'par', 'perp1', 'perp2',
        each shape (N_energy, N_theta, N_phi). Returned for diagnostic access.
    c_par_ts_raw : ndarray, shape (N_t,)
        Per-timestep parallel heating rate from direct velocity-space
        integration: sum_v q * v_par * E_par * delta_f * d3v.

    Raises
    ------
    ValueError
        If distribution data is empty, contains no finite values, or required
        arguments for fac_mode='instantaneous' are missing.
    '''

    if fac_mode == 'instantaneous':
        if b_xyz is None or b_times is None or v_ms_avg is None:
            raise ValueError(
                "fac_mode='instantaneous' requires b_xyz, b_times, and v_ms_avg."
            )

    q    = SPECIES[species]['q']
    mass = SPECIES[species]['m']

    t_dist = np.array(
        [(d['start_time'] + d['end_time']) * 0.5 for d in dist_chunk], dtype=float
    )
    energy = np.array([d['energy'] for d in dist_chunk], dtype=float)
    theta  = np.deg2rad(np.array([dist_chunk[0]['theta']], dtype=float))
    phi    = np.deg2rad(np.array([dist_chunk[0]['phi']],   dtype=float))

    if energy.size == 0:
        raise ValueError("Energy array is empty in this chunk.")

    # --- Optional photoelectron cut ---
    if ecut is not None:
        energy = np.where(energy > ecut, energy, np.nan)

    energy = np.where(energy > 0, energy, 1e-12)

    v = np.sqrt(2 * energy * np.abs(q) / mass)

    # --- Interpolate pre-filtered E onto this chunk's distribution times ---
    store_data('_e_filt_full', data={'x': e_filt_times, 'y': e_filt_y})
    tinterpol('_e_filt_full', t_dist, newname='_e_filt_chunk')
    e_interp = get_data('_e_filt_chunk')
    if e_interp is None:
        raise ValueError("Could not interpolate pre-filtered E field onto chunk times.")

    # --- Distribution fluctuations ---
    f = np.array([d['data'] * 1e12 for d in dist_chunk], dtype=float)
    if not np.any(np.isfinite(f)):
        raise ValueError("Distribution data in chunk contains no finite values.")
    mean_f = np.nanmean(f, axis=0)
    del_f  = f - mean_f

    # --- Velocity vectors in instrument frame (GSE), bulk-subtracted ---
    vx   = v * np.cos(theta[0]) * np.cos(phi[0]) - ve0[0]
    vy   = v * np.cos(theta[0]) * np.sin(phi[0]) - ve0[1]
    vz   = v * np.sin(theta[0])                  - ve0[2]
    vvec = np.stack([vx, vy, vz], axis=-1)   # (N_energy, N_theta, N_phi, 3)

    # =========================================================================
    # BRANCH: fixed FAC basis vs instantaneous FAC basis
    # =========================================================================

    if fac_mode == 'fixed':
        # --- Vectorised path: single basis applied to whole chunk ---
        epar    = np.dot(e_interp.y * 1e-3, eigen[0])   # (N_t,)
        eperp_1 = np.dot(e_interp.y * 1e-3, eigen[1])
        eperp_2 = np.dot(e_interp.y * 1e-3, eigen[2])

        vpar    = np.tensordot(vvec, eigen[0], axes=([-1], [0]))   # (N_e, N_th, N_ph)
        vperp_1 = np.tensordot(vvec, eigen[1], axes=([-1], [0]))
        vperp_2 = np.tensordot(vvec, eigen[2], axes=([-1], [0]))
        vperp   = np.sqrt(vperp_1**2 + vperp_2**2)

        # Per-timestep raw correlation: (N_t, N_e, N_th, N_ph)
        c_par_raw = q * vpar * epar[:, None, None, None] * del_f
        c_perp1_raw = q * vperp_1 * eperp_1[:, None, None, None] * del_f
        c_perp2_raw = q * vperp_2 * eperp_2[:, None, None, None] * del_f

        # Direct integration scalar with d3v -> (N_t,)
        c_par_ts_raw = np.nansum(c_par_raw * d3v[np.newaxis, ...], axis=(1, 2, 3))
        c_perp1_ts_raw = np.nansum(c_perp1_raw * d3v[np.newaxis, ...], axis=(1, 2, 3))
        c_perp2_ts_raw = np.nansum(c_perp2_raw * d3v[np.newaxis, ...], axis=(1, 2, 3))

        c_dict = {
            'par':   np.nanmean(c_par_raw,   axis=0),
            'perp1': np.nanmean(c_perp1_raw, axis=0),
            'perp2': np.nanmean(c_perp2_raw, axis=0),
        }

        c_ts_dict = {
            'par':   c_par_ts_raw,
            'perp1': c_perp1_ts_raw,
            'perp2': c_perp2_ts_raw,
        }

        # f retains its (N_t, N_e, N_th, N_ph) shape for _project_*
        f_for_proj  = f
        vpar_proj   = vpar
        vperp_proj  = vperp
        vperp1_proj = vperp_1
        vperp2_proj = vperp_2

    else:
        # --- Interpolate B onto all dist times at once ---
        b_on_dist = np.stack([
            np.interp(t_dist, b_times, b_xyz[:, i])
            for i in range(3)
        ], axis=1)   # (n_t, 3)

        # --- Build basis vectors, fully vectorised ---
        b_norms = np.linalg.norm(b_on_dist, axis=1, keepdims=True)
        e_par_t = b_on_dist / b_norms   # (n_t, 3)

        # --- Enforce orientation consistency relative to mean B ---
        mean_e_par = b_xyz.mean(axis=0)
        mean_e_par = mean_e_par / np.linalg.norm(mean_e_par)
        dots   = e_par_t @ mean_e_par           # (n_t,)
        signs  = np.sign(dots)[:, np.newaxis]   # (n_t, 1)
        e_par_t = e_par_t * signs               # flip to consistent hemisphere

        # --- Build perp vectors from orientation-corrected e_par ---
        v_rep     = np.broadcast_to(v_ms_avg, (len(t_dist), 3))
        bxv       = np.cross(e_par_t * b_norms * signs, v_rep)
        bxv_norm  = np.linalg.norm(bxv, axis=1, keepdims=True)
        e_perp2_t = bxv / bxv_norm

        bxvxb      = np.cross(bxv, e_par_t * b_norms * signs)
        bxvxb_norm = np.linalg.norm(bxvxb, axis=1, keepdims=True)
        e_perp1_t  = bxvxb / bxvxb_norm

        # --- Project E onto per-timestep basis ---
        e_vm     = e_interp.y * 1e-3               # (n_t, 3)
        epar_t   = np.einsum('ti,ti->t', e_vm, e_par_t)
        eperp1_t = np.einsum('ti,ti->t', e_vm, e_perp1_t)
        eperp2_t = np.einsum('ti,ti->t', e_vm, e_perp2_t)

        # --- Project v onto per-timestep basis ---
        # vvec is (N_t, N_e, N_th, N_ph, 3) because energy has a leading time
        # axis. Use einsum with ellipsis to contract only the trailing GSE axis,
        # producing (N_t, N_t_v, N_e, N_th, N_ph) — then we need to pair each
        # distribution timestep t with its own vvec[t]. Since the basis index t
        # and the vvec time index are the same axis, extract the diagonal:
        #   vpar_t[t] = vvec[t] · e_par_t[t]  ->  (N_t, N_e, N_th, N_ph)
        # The most efficient way is a batched einsum over the shared t axis:
        vpar_t    = np.einsum('teapx,tx->teap', vvec, e_par_t)    # (N_t, N_e, N_th, N_ph)
        vperp_1_t = np.einsum('teapx,tx->teap', vvec, e_perp1_t)
        vperp_2_t = np.einsum('teapx,tx->teap', vvec, e_perp2_t)

        # Diagnostic prints (retained from development; can be removed for production)
        print('epar_t mean:                  ', np.nanmean(epar_t))
        print('e_par_t.mean(axis=0):         ', e_par_t.mean(axis=0).round(4))
        print('dot(e_par_t_mean, eigen_par): ', np.dot(e_par_t.mean(axis=0), mean_e_par).round(6))
        print('e_interp.y mean (mV/m):       ', np.nanmean(e_interp.y, axis=0).round(6))

        # Per-timestep raw correlation: (N_t, N_e, N_th, N_ph)
        c_par_raw = q * vpar_t * epar_t[:, None, None, None] * del_f
        c_perp1_raw = q * vperp_1_t * eperp1_t[:, None, None, None] * del_f
        c_perp2_raw = q * vperp_2_t * eperp2_t[:, None, None, None] * del_f

        # Direct integration scalar with d3v -> (N_t,)
        c_par_ts_raw = np.nansum(c_par_raw * d3v[np.newaxis, ...], axis=(1, 2, 3))
        c_perp1_ts_raw = np.nansum(c_perp1_raw * d3v[np.newaxis, ...], axis=(1, 2, 3))
        c_perp2_ts_raw = np.nansum(c_perp2_raw * d3v[np.newaxis, ...], axis=(1, 2, 3))

        c_dict = {
            'par':   np.nanmean(c_par_raw,   axis=0),
            'perp1': np.nanmean(c_perp1_raw, axis=0),
            'perp2': np.nanmean(c_perp2_raw, axis=0),
        }

        c_ts_dict = {
            'par':   c_par_ts_raw,
            'perp1': c_perp1_ts_raw,
            'perp2': c_perp2_ts_raw,
        }
        
        # Mean velocity projections for binning — (N_e, N_th, N_ph)
        vpar_proj   = np.nanmean(vpar_t,    axis=0)
        vperp1_proj = np.nanmean(vperp_1_t, axis=0)
        vperp2_proj = np.nanmean(vperp_2_t, axis=0)
        vperp_proj  = np.sqrt(vperp1_proj**2 + vperp2_proj**2)

        # Add dummy time axis so _project_gyro/_project_cartesian nanmean(axis=0) works
        f_for_proj  = mean_f[np.newaxis, ...]
        vpar_proj   = vpar_proj[  np.newaxis, ...]
        vperp_proj  = vperp_proj[ np.newaxis, ...]
        vperp1_proj = vperp1_proj[np.newaxis, ...]
        vperp2_proj = vperp2_proj[np.newaxis, ...]

    # =========================================================================
    # PROJECT onto velocity-space grid
    # =========================================================================

    if projection == 'gyro':
        result, c_dict = _project_gyro(
            c_dict, vpar_proj, vperp_proj, f_for_proj,
            vth, vpar_edges, vperp_edges
        )
    elif projection == 'cartesian':
        result, c_dict = _project_cartesian(
            c_dict, vpar_proj, vperp1_proj, vperp2_proj, f_for_proj,
            vth, vpar_edges, vperp1_edges, vperp2_edges
        )
    else:
        raise ValueError(f"projection must be 'gyro' or 'cartesian', got '{projection}'")

    return result, c_dict, c_ts_dict


def field_particle_correlation(dist, e_field, b_field, bulkv, spintone=None,
                                cutoff=1, order=5, direction='parallel',
                                species='electron', counts_to_mask=0,
                                spacecraft_id=1, vpar_edges=None, vperp_edges=None,
                                vperp1_edges=None, vperp2_edges=None,
                                nbins=None, apply_filter=True, vmax=6,
                                n_subintervals=None, projection='gyro', ecut=None,
                                fac_mode='fixed'):
    '''
    Computes the field-particle correlation C(v) for all three FAC field
    components (E_par, E_perp1, E_perp2) simultaneously, binned onto a
    velocity-space grid normalised by the thermal velocity.

    All preprocessing (Lorentz transform, highpass filter, FAC basis, thermal
    velocity) is performed once on the full interval. The core correlation is
    then computed either over the full interval or split into n_subintervals
    time chunks, all sharing a consistent velocity grid.

    Two projections are supported:
    - 'gyro'      : 2D (v_par, |v_perp|) grid, assuming gyrotropy.
    - 'cartesian' : 3D (v_par, v_perp1, v_perp2) grid, no gyrotropy assumption.

    Two FAC modes are supported:
    - 'fixed'         : A single mean-field FAC basis is computed once from
                        mean(B) and applied to every timestep (default, fast).
    - 'instantaneous' : A new FAC basis is constructed at every distribution
                        timestep from the interpolated instantaneous B vector.
                        More physically appropriate for strongly turbulent
                        intervals (delta_B / B >= 0.5) but noticeably slower.

    Handles FPI interleave mode by processing alternate distributions separately
    and summing the results.

    Parameters
    ----------
    dist : list of dict
        List of particle distribution dicts, each containing keys:
        'start_time', 'end_time', 'energy', 'theta', 'phi', 'data'.
    e_field : str
        Tplot variable name for the electric field (mV/m).
    b_field : str
        Tplot variable name for the magnetic field (nT).
    bulkv : str
        Tplot variable name for the ion bulk velocity (km/s).
    spintone : str, optional
        Tplot variable name for the spintone correction (km/s).
    cutoff : float, optional
        Highpass filter cutoff frequency in Hz. Default is 1.
    order : int, optional
        Butterworth filter order. Default is 5.
    direction : str, optional
        Controls which temperature is used for vth normalisation.
        'parallel' uses temppara; 'perpendicular' uses tempperp.
        Default is 'parallel'.
    species : str, optional
        'ion' or 'electron'. Default is 'electron'.
    counts_to_mask : int, optional
        Minimum bin count threshold. Default is 0.
    spacecraft_id : int, optional
        MMS spacecraft ID (1-4). Default is 1.
    vpar_edges : ndarray, optional
        Pre-computed parallel velocity bin edges (vth-normalised).
    vperp_edges : ndarray, optional
        Pre-computed perpendicular speed bin edges (vth-normalised, gyro only).
    vperp1_edges : ndarray, optional
        Pre-computed v_perp1 bin edges (vth-normalised, cartesian only).
    vperp2_edges : ndarray, optional
        Pre-computed v_perp2 bin edges (vth-normalised, cartesian only).
    nbins : int, optional
        Number of bins along each velocity axis. Default is 60.
    apply_filter : bool, optional
        Whether to apply the highpass filter. Default is True.
    vmax : float, optional
        Velocity grid half-extent in units of vth. Default is 6.
    n_subintervals : int, optional
        If set, divides dist into this many time chunks and stacks results
        along a leading time axis. Default is None (full interval).
    projection : str, optional
        'gyro' or 'cartesian'. Default is 'gyro'.
    ecut : float or None, optional
        Low-energy cutoff in eV. Energies below this are masked before
        correlation. Default is None (no cut).
    fac_mode : str, optional
        'fixed'         — compute one mean-field FAC basis and apply it to all
                          timesteps (default, fast).
        'instantaneous' — compute a new FAC basis at every distribution
                          timestep from the instantaneous B vector. More
                          physically self-consistent for high-amplitude
                          turbulence (delta_B / B >= 0.5).

    Returns
    -------
    result : dict
        Full-interval mode — keys 'par', 'perp1', 'perp2', each containing:
            'c_binned' : ndarray, shape (nbins, nbins) or (nbins, nbins, nbins)
            'sumC'     : ndarray, same shape
        Plus shared keys:
            'counts'   : ndarray
            'f_binned' : ndarray
            'sumF'     : ndarray

        Time-resolved mode — same structure but 'c_binned' and 'f_binned'
        have a leading time axis, e.g. shape (n, nbins, nbins) for gyro.
        'sumC', 'counts', 'sumF' are None.

    c_dict : dict
        Full-interval: pre-binning arrays keyed by 'par', 'perp1', 'perp2',
        each shape (N_energy, N_theta, N_phi). For direct integration and
        diagnostic access without histogram averaging.
        Time-resolved: same keys, each shape (n_subintervals, N_e, N_th, N_ph).

    edges : dict
        Always contains 'vpar'. Gyro also contains 'vperp'. Cartesian also
        contains 'vperp1' and 'vperp2'.

    sub_times : ndarray or None
        Chunk centre times in time-resolved mode, else None.

    eigen : list of ndarray or None
        Fixed-mode: [e_par, e_perp1, e_perp2] unit vectors.
        Instantaneous-mode: None (basis varies per timestep).

    vth : float
        Thermal velocity in m/s.

    ve0 : ndarray, shape (3,)
        Mean bulk velocity in m/s used for frame shift.

    c_par_ts_full : ndarray, shape (N_t_total,)
        Full time series of per-distribution parallel heating rate from direct
        velocity-space integration: sum_v q * v_par * E_par * delta_f * d3v.

    Raises
    ------
    TypeError
        If tplot variable name arguments are not strings, or dist is not a list.
    ValueError
        If species, direction, projection, or fac_mode are invalid; inputs are
        empty or malformed; required tplot variables cannot be retrieved; or
        vmax is None.
    '''

    # --- Input validation ---
    if not isinstance(dist, list) or len(dist) == 0:
        raise ValueError("dist must be a non-empty list of distribution dicts.")
    for arg_name, arg_val in [('e_field', e_field), ('b_field', b_field), ('bulkv', bulkv)]:
        if not isinstance(arg_val, str):
            raise TypeError(f"'{arg_name}' must be a string, got {type(arg_val).__name__}")
    if spintone is not None and not isinstance(spintone, str):
        raise TypeError(f"'spintone' must be a string or None, got {type(spintone).__name__}")
    if species not in SPECIES:
        raise ValueError(f"species must be 'ion' or 'electron', got '{species}'")
    if direction not in ('parallel', 'perpendicular'):
        raise ValueError(f"direction must be 'parallel' or 'perpendicular', got '{direction}'")
    if not isinstance(spacecraft_id, (int, np.integer)) or spacecraft_id not in (1, 2, 3, 4):
        raise ValueError(f"spacecraft_id must be an integer 1-4, got {spacecraft_id}")
    if projection not in ('gyro', 'cartesian'):
        raise ValueError(f"projection must be 'gyro' or 'cartesian', got '{projection}'")
    if fac_mode not in ('fixed', 'instantaneous'):
        raise ValueError(f"fac_mode must be 'fixed' or 'instantaneous', got '{fac_mode}'")
    if vmax is None:
        raise ValueError("vmax must be set explicitly.")

    _nbins = nbins if nbins is not None else 60

    # =========================================================================
    # PREPROCESSING — performed once on the full interval
    # =========================================================================

    # lorentz() returns b_times as well as b_xyz (needed for instantaneous mode)
    smooth_name, b_xyz, b_times, v_ms_avg = lorentz(e_field, b_field, bulkv, spintone)

    if fac_mode == 'fixed':
        eigen = eigenvectors(b_xyz, v_ms_avg)
    else:
        # Basis built per-timestep inside _correlate_chunk; eigen is unused here
        eigen = None

    ve0 = v_ms_avg   # already in m/s, computed inside lorentz()

    e_smooth   = get_data(smooth_name)
    bulkv_data = get_data(bulkv)
    if bulkv_data is None:
        raise ValueError(f"Could not retrieve '{bulkv}'.")

    if apply_filter:
        sample_rate = 1 / np.median(np.diff(bulkv_data.times))
        e_filt_y    = highpass(e_smooth.y, cutoff, sample_rate, order)
    else:
        e_filt_y = e_smooth.y

    e_filt_times = e_smooth.times

    vth = _compute_vth(species, spacecraft_id, direction)
    if vth <= 0 or not np.isfinite(vth):
        raise ValueError(f"Thermal velocity is non-physical ({vth:.2e}).")

    # --- Native velocity-space volume element (d3v) ---
    # Computed from the first distribution's angular grid and the mean energy.
    # Shape: (N_energy, N_phi, N_theta)
    _theta_rad   = np.deg2rad(dist[0]['theta'])
    _phi_rad     = np.deg2rad(dist[0]['phi'])
    _energy_mean = np.nanmean(np.array([d['energy'] for d in dist]), axis=0)
    _q           = SPECIES[species]['q']
    _mass        = SPECIES[species]['m']
    _v0          = np.sqrt(2 * _energy_mean * abs(_q) / _mass)
    _dv          = np.abs(np.gradient(_v0,        axis=0))
    _dphi        = np.full_like(_phi_rad, np.deg2rad(360.0/32))
    _dtheta      = np.abs(np.gradient(_theta_rad, axis=2))
    d3v          = _v0**2 * np.cos(_theta_rad) * _dv * _dphi * _dtheta

    # --- Velocity grid edges ---
    if vpar_edges is None:
        vpar_edges = np.linspace(-vmax, vmax, _nbins + 1)
    if vperp_edges is None:
        vperp_edges = np.linspace(0, vmax, (_nbins // 2) + 1)
    if vperp1_edges is None:
        vperp1_edges = np.linspace(-vmax, vmax, _nbins + 1)
    if vperp2_edges is None:
        vperp2_edges = np.linspace(-vmax, vmax, _nbins + 1)

    edges = {
        'vpar':   vpar_edges,
        'vperp':  vperp_edges,
        'vperp1': vperp1_edges,
        'vperp2': vperp2_edges
    }

    # =========================================================================
    # INTERLEAVE DETECTION
    # =========================================================================

    is_interleaved = dist[0]['energy'].tolist() != dist[1]['energy'].tolist()

    def _run_on_subset(subset):
        '''Run _correlate_chunk on a subset of dists, handling interleave.'''

        chunk_kwargs = dict(
            e_filt_y       = e_filt_y,
            e_filt_times   = e_filt_times,
            eigen          = eigen,
            vth            = vth,
            ve0            = ve0,
            species        = species,
            counts_to_mask = counts_to_mask,
            vpar_edges     = vpar_edges,
            vperp_edges    = vperp_edges,
            vperp1_edges   = vperp1_edges,
            vperp2_edges   = vperp2_edges,
            nbins          = _nbins,
            projection     = projection,
            ecut           = ecut,
            d3v            = d3v,
            fac_mode       = fac_mode,
            b_xyz          = b_xyz,
            b_times        = b_times,
            v_ms_avg       = v_ms_avg,
        )

        if is_interleaved:
            r0, c_dict0, c_ts_dict0 = _correlate_chunk(list(subset[0::2]), **chunk_kwargs)
            r1, c_dict1, c_ts_dict1 = _correlate_chunk(list(subset[1::2]), **chunk_kwargs)

            counts   = r0['counts'] + r1['counts']
            sumF     = r0['sumF']   + r1['sumF']
            f_binned = sumF / np.where(counts > 0, counts, np.nan)
            merged   = {'counts': counts, 'f_binned': f_binned, 'sumF': sumF}

            for key in ('par', 'perp1', 'perp2'):
                sumC     = r0[key]['sumC'] + r1[key]['sumC']
                c_binned = sumC / np.where(counts > 0, counts, np.nan)
                merged[key] = {'c_binned': c_binned, 'sumC': sumC}

            # Average the pre-binning c_dict arrays across the two interleave sets
            c_dict_merged = {
                key: (c_dict0[key] + c_dict1[key]) / 2.0
                for key in ('par', 'perp1', 'perp2')
            }
            # Concatenate c_par_ts in time order (interleaved distributions are
            # already in chronological order within each subset)
            c_ts_dict_merged = {
                'par':   (c_ts_dict0['par'] + c_ts_dict1['par']) / 2.0,
                'perp1': (c_ts_dict0['perp1'] + c_ts_dict1['perp1']) / 2.0,
                'perp2': (c_ts_dict0['perp2'] + c_ts_dict1['perp2']) / 2.0,
            }

            return merged, c_dict_merged, c_ts_dict_merged
        else:
            return _correlate_chunk(list(subset), **chunk_kwargs)

    # =========================================================================
    # CHUNK LOOP — single chunk for full interval, n chunks for time-resolved
    # =========================================================================

    if n_subintervals is not None:
        n = int(n_subintervals)
        if n < 2:
            raise ValueError(f"n_subintervals must be >= 2, got {n}")
        if len(dist) < n:
            raise ValueError(
                f"n_subintervals ({n}) exceeds number of distributions ({len(dist)})"
            )
        chunks = np.array_split(dist, n)
    else:
        chunks = [dist]

    chunk_results  = []
    c_dict_list    = []
    c_par_ts_lists = []
    sub_times      = []

    c_ts_lists = {'par': [], 'perp1': [], 'perp2': []}
    
    for chunk in chunks:
        chunk = list(chunk)
        r, c_dict, c_ts_dict = _run_on_subset(chunk)
        chunk_results.append(r)
        c_dict_list.append(c_dict)
        for key in c_ts_lists:
            c_ts_lists[key].append(c_ts_dict[key])
        sub_times.append(
            np.mean([(d['start_time'] + d['end_time']) * 0.5 for d in chunk])
        )

    c_par_ts_full   = np.concatenate(c_ts_lists['par'])    # (N_t_total,)
    c_perp1_ts_full = np.concatenate(c_ts_lists['perp1'])  # (N_t_total,)
    c_perp2_ts_full = np.concatenate(c_ts_lists['perp2'])  # (N_t_total,)

    c_ts_dict_full = {
        'par':   c_par_ts_full,    # (N_t_total,),
        'perp1': c_perp1_ts_full,  # (N_t_total,),
        'perp2': c_perp2_ts_full,  # (N_t_total,),
    }
    sub_times     = np.array(sub_times)

    # =========================================================================
    # ASSEMBLE OUTPUT
    # =========================================================================

    if n_subintervals is None:
        return chunk_results[0], c_dict_list[0], edges, None, eigen, vth, ve0, c_ts_dict_full

    # Time-resolved — stack c_binned and f_binned along a leading time axis
    result = {
        'counts':   None,
        'sumF':     None,
        'f_binned': np.stack([r['f_binned'] for r in chunk_results], axis=0),
    }
    for key in ('par', 'perp1', 'perp2'):
        result[key] = {
            'c_binned': np.stack([r[key]['c_binned'] for r in chunk_results], axis=0),
            'sumC':     None,
        }

    c_dict_ts = {
        key: np.stack([cd[key] for cd in c_dict_list], axis=0)
        for key in ('par', 'perp1', 'perp2')
    }

    return result, c_dict_ts, edges, sub_times, eigen, vth, ve0, c_ts_dict_full
