import numpy as np

def current_density(n, vp, ve):

  '''
  Returns j as a 4D array with dimensions (x, y, z, magnitude) calculated by j = n * q * (vp-ve). Assumes all products have been interpolated/smoothed onto same
  cadence.

  Parameters
  ----------

  n : array-like
    FPI numberdensity level 2 product. Units in cm^3. Quasineutrality is assumed

  vp : array-like
    FPI proton bulkv level 2 product. Units in km/s. Shape (N, 3)

  ve : array-like
    FPI electron bulkv level 2 product. Units in km/s. Shape (N, 3)

  Returns
  -------
  j : ndarray
      Current density with with shape (N, 4) where columns are [jx, jy, jz, |j|].
        Units in A/m^2

  Raises
  ------
  ValueError
      If inputs have incompatible shapes or invalid values
  TypeError
      If inputs cannot be converted to numpy arrays
    
  Notes
  -----
  Assumes quasi-neutrality: n_proton ≈ n_electron ≈ n
    
  '''

  # Input validation and conversion
  try:
      n = np.asarray(n, dtype=float)
      vp = np.asarray(vp, dtype=float)
      ve = np.asarray(ve, dtype=float)
  except (ValueError, TypeError) as e:
      raise TypeError(f"Could not convert inputs to numpy arrays: {e}")
    
  # Check for empty arrays
  if n.size == 0:
      raise ValueError("n (number density) cannot be empty")
  if vp.size == 0:
      raise ValueError("vp (proton velocity) cannot be empty")
  if ve.size == 0:
      raise ValueError("ve (electron velocity) cannot be empty")
    
  # Ensure n is 1D
  if n.ndim != 1:
      raise ValueError(f"n must be 1D array, got {n.ndim}D array with shape {n.shape}")
    
  # Ensure velocities are 2D with 3 components
  if vp.ndim == 1:
      if len(vp) == 3:
          vp = vp.reshape(1, 3)
      else:
          raise ValueError(
              f"vp must be 2D array with shape (N, 3) or 1D array with 3 components, "
              f"got 1D array with {len(vp)} elements"
            )
  elif vp.ndim != 2:
      raise ValueError(f"vp must be 2D array, got {vp.ndim}D array")
    
  if ve.ndim == 1:
      if len(ve) == 3:
          ve = ve.reshape(1, 3)
      else:
        raise ValueError(
              f"ve must be 2D array with shape (N, 3) or 1D array with 3 components, "
              f"got 1D array with {len(ve)} elements"
            )
  elif ve.ndim != 2:
      raise ValueError(f"ve must be 2D array, got {ve.ndim}D array")
    
  # Check velocity components
  if vp.shape[1] != 3:
      raise ValueError(
          f"vp must have 3 velocity components (x, y, z), got {vp.shape[1]} components"
        )
  if ve.shape[1] != 3:
      raise ValueError(
          f"ve must have 3 velocity components (x, y, z), got {ve.shape[1]} components"
        )
    
  # Check dimension compatibility
  if len(n) != vp.shape[0]:
      raise ValueError(
          f"Length of n ({len(n)}) must match first dimension of vp ({vp.shape[0]})"
        )
  if len(n) != ve.shape[0]:
      raise ValueError(
          f"Length of n ({len(n)}) must match first dimension of ve ({ve.shape[0]})"
        )
  if vp.shape != ve.shape:
      raise ValueError(
          f"vp and ve must have same shape, got vp: {vp.shape}, ve: {ve.shape}"
        )
    
  # Check for physical validity
  if np.any(n < 0):
      raise ValueError("Number density n cannot contain negative values")
  if np.any(~np.isfinite(n)):
      raise ValueError("Number density n contains NaN or Inf values")
  if np.any(~np.isfinite(vp)):
      raise ValueError("Proton velocity vp contains NaN or Inf values")
  if np.any(~np.isfinite(ve)):
      raise ValueError("Electron velocity ve contains NaN or Inf values")
    
  # Warn if density is suspiciously high or low (optional, can be removed if too strict)
  if np.any(n > 1e8):  # > 10^8 cm^-3 is very high for magnetosphere
      import warnings
      warnings.warn(
          f"Density contains very high values (max: {np.max(n):.2e} cm^-3). "
          "Check if units are correct.",
          UserWarning
        )
    
  # Physical constants
  q = 1.60217663e-19  # Elementary charge in C

  # Unit conversions
  n = n * 1e6 # cm^3 to m^3

  vp = vp * 1e3 # km/s to m/s

  ve = ve * 1e3 # km/s to m/s

  # Calculate current density vector: j = n * q * (vp - ve)
  j_vec = n[:, None] * q * (vp - ve)

  # Calculate magnitude
  j_mag = np.linalg.norm(j_vec, axis=1)

  # Combine vector components and magnitude
  j = np.hstack([j_vec, j_mag[:, None]])

  return j

  
