import numpy as np

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
        ))

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
