import numpy as np
from pyspedas import get_data, store_data, tinterpol, subtract
import scipy.signal

def boxcar_averager(data_times, data, interp_times):
  '''
  Boxcar averages data onto interp_times using the neareast timestamps in data_times,
  with window width determined by interp_times / data_times.

  Parameters
  ----------
  data_times : array-like
      Timestamps corresponding to data points
  data : array-like
      Data array to be averaged (first axis should match length of data_times)
  interp_times : array-like
      Target timestamps for interpolated/averaged output

  Returns
  -------
  interp_times : ndarray
    Input interpolation times (returned for convenience)
  averaged : ndarray
    Boxcar averaged data at interp_times

  Raises
  ------
  ValueError
    If inputs are invalid (empty, mismatched dimensions, non-monotonic)
  TypeError
    If inputs cannot be converted to numpy arrays
  '''

  # Input validation and conversion
  try:

    data_times = np.asarray(data_times)
    data = np.asarray(data)
    interp_times = np.asarray(interp_times)

  except (ValueError, TypeError) as e:

    raise TypeError(f"Could not convert inputs to numpy arrays: {e}")

  # Check dimensions
  if data_times.ndim != 1:

      raise ValueError(f"data_times must be 1D, got {data_times.ndim}D")

  if interp_times.ndim != 1:

      raise ValueError(f"interp_times must be 1D, got {interp_times.ndim}D")

  if data.shape[0] != len(data_times):

      raise ValueError(
          f"First dimension of data ({data.shape[0]}) must match "
          f"length of data_times ({len(data_times)})"
        )
    
  # Check for sufficient data points
  if len(data_times) < 2:

      raise ValueError("data_times must have at least 2 points to compute differences")

  if len(interp_times) < 2:

      raise ValueError("interp_times must have at least 2 points to compute differences")
    
  # Check for monotonicity (important for searchsorted)
  if not np.all(np.diff(data_times) > 0):

      raise ValueError("data_times must be strictly monotonically increasing")

  if not np.all(np.diff(interp_times) > 0):

      raise ValueError("interp_times must be strictly monotonically increasing")

  # Compute time steps
  dt1 = np.median(np.diff(data_times))
  dt2 = np.median(np.diff(interp_times))

  # Compute window width
  width = int(round(dt2 / dt1))

  if width < 1:

      raise ValueError(
          f"Computed window width ({width}) is less than 1. "
          f"interp_times spacing (dt2={dt2:.3e}) may be too small "
          f"compared to data_times spacing (dt1={dt1:.3e})"
        )

  half = width // 2

  # Find nearest indices
  idx1 = np.searchsorted(data_times, interp_times)

  idx1 = np.clip(idx1, 0, len(data_times) - 1)

  # Initialize output array
  averaged = np.zeros((len(interp_times),) + data.shape[1:])

  # Perform averaging
  for i, centre_idx in enumerate(idx1):

    start = max(centre_idx - half, 0)
    end = min(centre_idx + half + 1, len(data))

    # Additional check for empty window (shouldn't happen, but defensive)
    if end <= start:
      raise RuntimeError(
        f"Empty averaging window at index {i}: start={start}, end={end}"
            )

    averaged[i] = np.mean(data[start:end], axis = 0)

  return interp_times, averaged

def lorentz(e_field, b_field, bulkv, spintone=None):
    '''
    Transforms the electric field from the spacecraft frame to the plasma frame
    using the Lorentz transformation: E_plasma = E_sc + (v x B).

    Parameters
    ----------
    e_field : str
        Tplot variable name for the electric field in the spacecraft frame (mV/m).
    b_field : str
        Tplot variable name for the magnetic field in the spacecraft frame (nT).
    bulkv : str
        Tplot variable name for the ion bulk velocity (km/s). Used to set the
        output cadence via boxcar averaging.
    spintone : str, optional
        Tplot variable name for the spintone product (km/s). If provided, it is
        subtracted from bulkv before the transform. Default is None.

    Returns
    -------
    smooth_name : str
        Tplot variable name of the Lorentz-transformed and smoothed E field,
        stored as 'e_field_transformed_smooth'.

    Raises
    ------
    TypeError
        If any tplot variable name argument is not a string.
    ValueError
        If any tplot variable cannot be retrieved, has wrong shape, or contains
        non-finite values.
    '''

    # --- Input type validation ---
    for arg_name, arg_val in [('e_field', e_field), ('b_field', b_field), ('bulkv', bulkv)]:
        if not isinstance(arg_val, str):
            raise TypeError(f"'{arg_name}' must be a string tplot variable name, got {type(arg_val).__name__}")
    if spintone is not None and not isinstance(spintone, str):
        raise TypeError(f"'spintone' must be a string tplot variable name or None, got {type(spintone).__name__}")

    # --- Retrieve tplot variables ---
    e_sc = get_data(e_field)
    if e_sc is None:
        raise ValueError(f"Could not retrieve tplot variable '{e_field}'. Check it has been loaded.")

    b_sc = get_data(b_field)
    if b_sc is None:
        raise ValueError(f"Could not retrieve tplot variable '{b_field}'. Check it has been loaded.")

    bulkv_data = get_data(bulkv)
    if bulkv_data is None:
        raise ValueError(f"Could not retrieve tplot variable '{bulkv}'. Check it has been loaded.")

    # --- Validate E field shape ---
    if e_sc.y.ndim != 2 or e_sc.y.shape[1] < 3:
        raise ValueError(
            f"'{e_field}' must be a 2D array with at least 3 components (Ex, Ey, Ez), "
            f"got shape {e_sc.y.shape}"
        )

    # --- Validate B field shape ---
    if b_sc.y.ndim != 2 or b_sc.y.shape[1] < 3:
        raise ValueError(
            f"'{b_field}' must be a 2D array with at least 3 components (Bx, By, Bz), "
            f"got shape {b_sc.y.shape}"
        )

    # --- Interpolate B onto E cadence ---
    tinterpol(b_field, e_sc.times, newname='b_interp')
    b_interp_data = get_data('b_interp')
    if b_interp_data is None:
        raise ValueError("Interpolation of B field failed — 'b_interp' could not be retrieved after tinterpol.")

    # --- Handle spintone subtraction ---
    if spintone is not None:
        spintone_data = get_data(spintone)
        if spintone_data is None:
            raise ValueError(f"Could not retrieve spintone variable '{spintone}'. Check it has been loaded.")
        subtract(bulkv, spintone, newname='bulkv_corrected')
        v_corrected_data = get_data('bulkv_corrected')
        if v_corrected_data is None:
            raise ValueError("Spintone subtraction failed — 'bulkv_corrected' could not be retrieved.")
        v_ms_avg = np.mean(v_corrected_data.y, axis=0) * 1e3  # km/s to m/s
    else:
        v_ms_avg = np.mean(bulkv_data.y, axis=0) * 1e3  # km/s to m/s

    # --- Validate averaged velocity ---
    if v_ms_avg.shape != (3,):
        raise ValueError(
            f"Averaged bulk velocity must have 3 components (vx, vy, vz), got shape {v_ms_avg.shape}"
        )
    if not np.all(np.isfinite(v_ms_avg)):
        raise ValueError("Averaged bulk velocity contains NaN or Inf values. Check input data quality.")

    # --- Validate interpolated B ---
    b_xyz = b_interp_data.y[:, 0:3]
    if not np.all(np.isfinite(b_xyz)):
        raise ValueError("Interpolated B field contains NaN or Inf values. Check input data quality.")

    # --- Validate E field values ---
    if not np.all(np.isfinite(e_sc.y)):
        raise ValueError(f"'{e_field}' contains NaN or Inf values. Check input data quality.")

    # --- Lorentz transform: E_plasma = E_sc + (v x B) ---
    # E: mV/m -> V/m (*1e-3), v: already in m/s, B: nT -> T (*1e-9), result: V/m -> mV/m (*1e3)
    e_lorentz = (e_sc.y * 1e-3) + np.cross(v_ms_avg, b_xyz * 1e-9)  # V/m
    e_lorentz_mvm = e_lorentz * 1e3  # back to mV/m for storage

    # --- Store and smooth ---
    store_data('e_field_transformed', data={'x': e_sc.times, 'y': e_lorentz_mvm})
    transformed_data = get_data('e_field_transformed')
    if transformed_data is None:
        raise ValueError("Failed to store or retrieve 'e_field_transformed'.")

    smooth_times, smooth_y = boxcar_averager(transformed_data.times, transformed_data.y, bulkv_data.times)

    smooth_name = 'e_field_transformed_smooth'
    store_data(smooth_name, data={'x': bulkv_data.times, 'y': smooth_y})

    if get_data(smooth_name) is None:
        raise ValueError(f"Failed to store smoothed output as '{smooth_name}'.")

    return smooth_name, b_xyz, v_ms_avg

def eigenvectors(b_xyz, v_ms_avg):
    '''
    Computes orthonormal basis vectors for field-aligned coordinates (FAC) from
    the mean magnetic field and ion bulk velocity.

    e_par  : parallel to mean B
    e_perp2: parallel to (B x v), i.e. the "out of plane" direction
    e_perp1: completes the right-handed set, lying in the B-v plane

    Parameters
    ----------
    b_xyz : array-like
        Magnetic field time series, shape (N, 3), in any consistent units.
    v_ms_avg : array-like
        Mean ion bulk velocity vector, shape (3,), in any consistent units.

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
        degenerate cross products (e.g. v parallel to B).
    '''

    # --- Input conversion ---
    try:
        b_xyz = np.asarray(b_xyz, dtype=float)
        v_ms_avg = np.asarray(v_ms_avg, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {e}")

    # --- Shape validation ---
    if b_xyz.ndim != 2 or b_xyz.shape[1] != 3:
        raise ValueError(
            f"b_xyz must have shape (N, 3), got {b_xyz.shape}"
        )
    if v_ms_avg.shape != (3,):
        raise ValueError(
            f"v_ms_avg must have shape (3,), got {v_ms_avg.shape}"
        )

    # --- Finite value checks ---
    if not np.all(np.isfinite(b_xyz)):
        raise ValueError("b_xyz contains NaN or Inf values. Check input data quality.")
    if not np.all(np.isfinite(v_ms_avg)):
        raise ValueError("v_ms_avg contains NaN or Inf values. Check input data quality.")

    # --- Compute mean field and e_par ---
    b0 = np.nanmean(b_xyz, axis=0)
    b0_norm = np.linalg.norm(b0)
    if b0_norm < 1e-10:
        raise ValueError(
            f"Mean magnetic field magnitude is effectively zero ({b0_norm:.2e}). "
            "Cannot define a parallel direction."
        )
    e_par = b0 / b0_norm

    # --- Check v is not parallel to B (would give degenerate cross products) ---
    v_norm = np.linalg.norm(v_ms_avg)
    if v_norm < 1e-10:
        raise ValueError(
            f"v_ms_avg magnitude is effectively zero ({v_norm:.2e}). "
            "Cannot define perpendicular directions."
        )
    cos_angle = np.dot(b0, v_ms_avg) / (b0_norm * v_norm)
    if abs(cos_angle) > 1 - 1e-6:
        raise ValueError(
            "v_ms_avg is parallel (or anti-parallel) to mean B. "
            "Cannot define unique perpendicular directions."
        )

    # --- Compute perpendicular basis vectors ---
    bxv = np.cross(b0, v_ms_avg)          # B x v (e_perp2 direction)
    bxv_norm = np.linalg.norm(bxv)
    e_perp2 = bxv / bxv_norm

    bxvxb = np.cross(bxv, b0)             # (B x v) x B (e_perp1 direction)
    bxvxb_norm = np.linalg.norm(bxvxb)
    e_perp1 = bxvxb / bxvxb_norm

    eigen = [e_par, e_perp1, e_perp2]
    return eigen

def highpass(data, cutoff, sample_rate, order):
    '''
    Applies a zero-phase Butterworth highpass filter to data.

    Parameters
    ----------
    data : array-like
        Input data to filter. Filtering is applied along axis=0 (time axis).
    cutoff : float
        Cutoff frequency in Hz.
    sample_rate : float
        Sampling frequency of the data in Hz.
    order : int
        Order of the Butterworth filter.

    Returns
    -------
    filtered_data : ndarray
        Highpass filtered data, same shape as input.

    Raises
    ------
    TypeError
        If inputs cannot be converted to expected types.
    ValueError
        If filter parameters are invalid or data is too short to filter.
    '''

    # --- Input conversion and validation ---
    try:
        data = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert data to numpy array: {e}")

    try:
        cutoff = float(cutoff)
        sample_rate = float(sample_rate)
    except (ValueError, TypeError) as e:
        raise TypeError(f"cutoff and sample_rate must be numeric: {e}")

    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError(f"order must be a positive integer, got {order}")

    # --- Physical parameter checks ---
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if cutoff >= sample_rate / 2:
        raise ValueError(
            f"cutoff ({cutoff} Hz) must be less than the Nyquist frequency "
            f"({sample_rate / 2} Hz)."
        )

    # --- Data checks ---
    if data.size == 0:
        raise ValueError("data cannot be empty.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data contains NaN or Inf values. Check input data quality.")

    # sosfiltfilt requires at least 3 * (2 * order) + 1 samples along filter axis
    min_samples = 3 * (2 * order) + 1
    if data.shape[0] < min_samples:
        raise ValueError(
            f"data has too few samples ({data.shape[0]}) for filter order {order}. "
            f"Minimum required: {min_samples}."
        )

    # --- Filter ---
    sos = scipy.signal.butter(order, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data, axis=0)

    return filtered_data


SPECIES = {
    'ion': {
        'q': 1.60217663e-19,
        'm': 1.67262192e-27,
        'label': 'ion'
    },
    'electron': {
        'q': -1.60217663e-19,
        'm': 9.10938371e-31,
        'label': 'electron'
    }
}

def _compute_vth(dist, species, spacecraft_id, direction):
    q = SPECIES[species]['q']
    mass = SPECIES[species]['m']
    inst = 'dis' if species == 'ion' else 'des'
    temp_dir = 'para' if direction == 'parallel' else 'perp'
    temp_tvar = f'mms{spacecraft_id}_{inst}_temp{temp_dir}_brst'
    temp_data = get_data(temp_tvar)
    if temp_data is None:
        raise ValueError(f"Could not retrieve temperature variable '{temp_tvar}'.")
    return np.sqrt(2 * np.abs(q) * np.mean(temp_data.y) / mass)


def _compute_vnorm(dist, bulkv, species, vth, eigen):
    q = SPECIES[species]['q']
    mass = SPECIES[species]['m']
    energy = np.array([d['energy'] for d in dist], dtype=float)
    theta = np.deg2rad(np.array([dist[0]['theta']], dtype=float))
    phi = np.deg2rad(np.array([dist[0]['phi']], dtype=float))
    energy = np.where(energy > 0, energy, 1e-12)
    v = np.sqrt(2 * energy * np.abs(q) / mass)

    bulkv_data = get_data(bulkv)
    ve0 = np.mean(bulkv_data.y, axis=0) * 1e3
    vx = v * np.cos(theta[0]) * np.cos(phi[0]) - ve0[0]
    vy = v * np.cos(theta[0]) * np.sin(phi[0]) - ve0[1]
    vz = v * np.sin(theta[0])                  - ve0[2]
    vvec = np.stack([vx, vy, vz], axis=-1)

    vpar   = np.tensordot(vvec, eigen[0], axes=([-1], [0]))
    vperp1 = np.tensordot(vvec, eigen[1], axes=([-1], [0]))
    vperp2 = np.tensordot(vvec, eigen[2], axes=([-1], [0]))
    vperp  = np.sqrt(vperp1**2 + vperp2**2)

    vpar_n  = np.nanmean(vpar,  axis=0).ravel() / vth
    vperp_n = np.nanmean(vperp, axis=0).ravel() / vth
    return vpar_n, vperp_n

def field_particle_correlation(dist, e_field, b_field, bulkv, spintone=None,
                                cutoff=1, order=5, direction='parallel',
                                species='electron', counts_to_mask=0,
                                spacecraft_id=1, vpar_edges=None, vperp_edges=None,
                                nbins=None, apply_filter=True, vmax=None):
    '''
    Computes the field-particle correlation C(v_par, v_perp) for a given
    particle species and field direction, binned onto a 2D velocity-space grid
    normalised by the thermal velocity.

    Handles FPI interleave mode by recursively processing alternate distributions
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
        Correlation direction, either 'parallel' or 'perpendicular'. Default is 'parallel'.
    species : str, optional
        Particle species, either 'ion' or 'electron'. Default is 'electron'.
    counts_to_mask : int, optional
        Minimum bin count threshold below which bins are masked as NaN. Default is 0.
    spacecraft_id : int, optional
        MMS spacecraft ID (1-4). Default is 1.
    vpar_edges : ndarray, optional
        Pre-computed parallel velocity bin edges normalised by vth, shape (nbins+1,).
        If provided, overrides internal edge calculation. Used internally by the
        interleave branch to enforce a consistent grid across both distribution subsets.
        Default is None.
    vperp_edges : ndarray, optional
        Pre-computed perpendicular velocity bin edges normalised by vth, shape (nbins+1,).
        If provided, overrides internal edge calculation. Used internally by the
        interleave branch to enforce a consistent grid across both distribution subsets.
        Default is None.

    Returns
    -------
    c_binned : ndarray
        2D array of binned correlation values in (v_par, v_perp) space,
        normalised by thermal velocity. Bins below counts_to_mask are NaN.
        Shape (nbins, nbins).
    sumC : ndarray
        2D array of summed (unmasked) correlation values before normalisation
        by bin counts. Returned for use in interleave accumulation.
        Shape (nbins, nbins).
    counts : ndarray
        2D array of particle counts per bin. Returned for use in interleave
        accumulation. Shape (nbins, nbins).
    vpar_edges : ndarray
        Parallel velocity bin edges normalised by vth, shape (nbins+1,).
    vperp_edges : ndarray
        Perpendicular velocity bin edges normalised by vth, shape (nbins+1,).

    Raises
    ------
    TypeError
        If tplot variable name arguments are not strings, or dist is not a list.
    ValueError
        If species or direction are invalid, inputs are empty or malformed,
        required tplot variables cannot be retrieved, or nbins is too small.
    '''

    # --- Input validation ---
    if not isinstance(dist, list) or len(dist) == 0:
        raise ValueError("dist must be a non-empty list of distribution dicts.")
    for arg_name, arg_val in [('e_field', e_field), ('b_field', b_field), ('bulkv', bulkv)]:
        if not isinstance(arg_val, str):
            raise TypeError(f"'{arg_name}' must be a string tplot variable name, got {type(arg_val).__name__}")
    if spintone is not None and not isinstance(spintone, str):
        raise TypeError(f"'spintone' must be a string or None, got {type(spintone).__name__}")
    if species not in SPECIES:
        raise ValueError(f"species must be 'ion' or 'electron', got '{species}'")
    if direction not in ('parallel', 'perpendicular'):
        raise ValueError(f"direction must be 'parallel' or 'perpendicular', got '{direction}'")
    if not isinstance(spacecraft_id, (int, np.integer)) or spacecraft_id not in (1, 2, 3, 4):
        raise ValueError(f"spacecraft_id must be an integer 1-4, got {spacecraft_id}")

    # --- Species constants ---
    q = SPECIES[species]['q']
    mass = SPECIES[species]['m']

    # --- Distribution arrays ---
    t_dist = np.array([(d['start_time'] + d['end_time']) * 0.5 for d in dist], dtype=float)
    energy = np.array([d['energy'] for d in dist], dtype=float)
    theta = np.deg2rad(np.array([dist[0]['theta']], dtype=float))
    phi = np.deg2rad(np.array([dist[0]['phi']], dtype=float))

    if energy.size == 0:
        raise ValueError("Energy array is empty. Check distribution dicts.")

    energy = np.where(energy > 0, energy, 1e-12)

    # --- Velocity in instrument frame, then subtract bulk velocity ---
    v = np.sqrt(2 * energy * np.abs(q) / mass)

    bulkv_data = get_data(bulkv)
    if bulkv_data is None:
        raise ValueError(f"Could not retrieve tplot variable '{bulkv}'.")
    ve0 = np.mean(bulkv_data.y, axis=0) * 1e3  # km/s to m/s
    if ve0.shape != (3,):
        raise ValueError(f"Bulk velocity must have 3 components, got shape {ve0.shape}")

    vx = v * np.cos(theta[0]) * np.cos(phi[0]) - ve0[0]
    vy = v * np.cos(theta[0]) * np.sin(phi[0]) - ve0[1]
    vz = v * np.sin(theta[0])                  - ve0[2]
    vvec = np.stack([vx, vy, vz], axis=-1)

    # --- Lorentz transform and FAC basis ---
    smooth_name, b_xyz, v_ms_avg = lorentz(e_field, b_field, bulkv, spintone)
    eigen = eigenvectors(b_xyz, v_ms_avg)

    # --- Interleave mode: recurse on alternating distributions ---
    if dist[0]['energy'].tolist() != dist[1]['energy'].tolist():

      vth = _compute_vth(dist, species, spacecraft_id, direction)

      if vmax is not None:
          _nbins = nbins if nbins is not None else 60
          vpar_edges  = np.linspace(-vmax, vmax, _nbins + 1)
          vperp_edges = np.linspace(0,     vmax, _nbins + 1)
      else:
          vpar_n, vperp_n = _compute_vnorm(dist, bulkv, species, vth, eigen)
          _nbins = nbins if nbins is not None else int(((vpar_n.max() - vpar_n.min()) * 100) // 10) + 20
          vpar_edges  = np.linspace(vpar_n.min(), vpar_n.max(), _nbins + 1)
          vperp_edges = np.linspace(0, vperp_n.max(), _nbins + 1)

      _, sumc1, counts1, _, sumF1, _ = field_particle_correlation(
          dist[0::2], e_field, b_field, bulkv, spintone,
          cutoff, order, direction, species, counts_to_mask, spacecraft_id,
          vpar_edges=vpar_edges, vperp_edges=vperp_edges, nbins=_nbins, vmax=vmax
      )
      _, sumc2, counts2, _, sumF2, _ = field_particle_correlation(
          dist[1::2], e_field, b_field, bulkv, spintone,
          cutoff, order, direction, species, counts_to_mask, spacecraft_id,
          vpar_edges=vpar_edges, vperp_edges=vperp_edges, nbins=_nbins, vmax=vmax
      )
      sumc = sumc1 + sumc2
      sumf = sumF1 + sumF2
      counts = counts1 + counts2
      c_binned = np.full_like(sumc, np.nan, dtype=float)
      f_binned = np.full_like(sumf, np.nan, dtype=float)
      mask = counts > counts_to_mask
      c_binned[mask] = sumc[mask] / counts[mask]
      return c_binned, sumc, counts, vpar_edges, vperp_edges, sumf, f_binned

    vpar    = np.tensordot(vvec, eigen[0], axes=([-1], [0]))
    vperp_1 = np.tensordot(vvec, eigen[1], axes=([-1], [0]))
    vperp_2 = np.tensordot(vvec, eigen[2], axes=([-1], [0]))

    # --- Distribution fluctuations ---
    f = np.array([d['data'] * 1e12 for d in dist], dtype=float)
    if not np.any(np.isfinite(f)):
        raise ValueError("Distribution data contains no finite values.")
    mean_f = np.nanmean(f, axis=0)
    del_f = f - mean_f

    # --- Highpass filter E field ---
    e = get_data(smooth_name)
    if e is None:
        raise ValueError(f"Could not retrieve smoothed E field '{smooth_name}'.")
      
    if apply_filter:
      sample_rate = 1 / np.median(np.diff(bulkv_data.times))
      e_filt = highpass(e.y, cutoff, sample_rate, order)
      store_data('e_filt', data={'x': e.times, 'y': e_filt})
      tinterpol('e_filt', t_dist, newname='e_filt_interp')

    else:
      tinterpol(smooth_name, t_dist, newname='e_filt_interp')
      
    e_interp = get_data('e_filt_interp')
    if e_interp is None:
        raise ValueError("Could not retrieve interpolated E field 'e_filt_interp'.")

    # --- Project E onto FAC basis ---
    epar    = np.dot(e_interp.y * 1e-3, eigen[0])
    eperp_1 = np.dot(e_interp.y * 1e-3, eigen[1])
    eperp_2 = np.dot(e_interp.y * 1e-3, eigen[2])

    # --- Correlation ---
    if direction == 'parallel':
        c = q * vpar * epar[:, None, None, None] * del_f
    elif direction == 'perpendicular':
        c1 = q * vperp_1 * eperp_1[:, None, None, None] * del_f
        c2 = q * vperp_2 * eperp_2[:, None, None, None] * del_f
        c = c1 + c2

    c = np.nanmean(c, axis=0)

    # --- Thermal velocity ---
    inst = 'dis' if species == 'ion' else 'des'
    temp_dir = 'para' if direction == 'parallel' else 'perp'
    temp_tvar = f'mms{spacecraft_id}_{inst}_temp{temp_dir}_brst'
    temp_data = get_data(temp_tvar)
    if temp_data is None:
        raise ValueError(
            f"Could not retrieve temperature variable '{temp_tvar}'. "
            "Check it has been loaded."
        )
    vth = np.sqrt(2 * np.abs(q) * np.mean(temp_data.y) / mass)

    if vth <= 0 or not np.isfinite(vth):
        raise ValueError(f"Thermal velocity is non-physical ({vth:.2e}). Check temperature data.")

    # --- Flatten and mask ---
    vperp = np.sqrt(vperp_1 ** 2 + vperp_2 ** 2)

    f_flat     = f.ravel()                              
    c_flat     = c.ravel()
    vpar_flat  = np.nanmean(vpar, axis=0).ravel()
    vperp_flat = np.nanmean(vperp, axis=0).ravel()

    finite_mask = np.isfinite(c_flat) & np.isfinite(vpar_flat) & np.isfinite(vperp_flat)
    c_flat     = c_flat[finite_mask]
    vpar_flat  = vpar_flat[finite_mask]
    vperp_flat = vperp_flat[finite_mask]

    if f_flat.size == 0:
        raise ValueError("Distribution has 0 values. Check input data")

    if c_flat.size == 0:
        raise ValueError("No finite correlation values remain after masking. Check input data.")

    # --- Normalise by thermal velocity ---
    vpar_n  = vpar_flat  / vth
    vperp_n = vperp_flat / vth

    # --- Bin edges ---
    if nbins is None:
      if vmax is not None:
          nbins = 60
      else:
          if direction == 'parallel':
              nbins = int(((vpar_n.max() - vpar_n.min()) * 100) // 10) + 20
          else:
              nbins = int(((vperp_n.max() - vperp_n.min()) * 100) // 10) + 20
    if nbins < 2:
        raise ValueError(f"Computed nbins ({nbins}) is too small.")

    if vpar_edges is None:
        if vmax is not None:
            vpar_edges  = np.linspace(-vmax, vmax,  nbins + 1)
            vperp_edges = np.linspace(0,     vmax,  nbins + 1)
        else:
            vpar_edges  = np.linspace(vpar_n.min(),  vpar_n.max(),  nbins + 1)
            vperp_edges = np.linspace(0,             vperp_n.max(), nbins + 1)

    # --- 2D histogram ---
    sumC, _, _   = np.histogram2d(vpar_n, vperp_n, bins=[vpar_edges, vperp_edges], weights=c_flat)
    counts, _, _ = np.histogram2d(vpar_n, vperp_n, bins=[vpar_edges, vperp_edges])

    c_binned = np.full_like(sumC, np.nan, dtype=float)
    mask = counts > counts_to_mask
    c_binned[mask] = sumC[mask] / counts[mask]

    # --- 2D Distribution Histogram ---
    sumF, _, _ = np.histogram2d(vpar_n, vperp_n, bins=[vpar_edges, vperp_edges], weights=f_flat)
                                  
    f_binned = np.full_like(sumF, np.nan, dtype=float)

    return c_binned, sumC, counts, vpar_edges, vperp_edges, sumF, f_binned
