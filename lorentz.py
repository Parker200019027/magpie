import numpy as np
from pyspedas import get_data, store_data, tinterpol, subtract


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
