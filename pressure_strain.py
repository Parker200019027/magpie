import numpy as np


def pressure_strain(prestensor, positions, bulkv_arr):
    '''
    Returns Pi-D and pressure dilatation calculated from MMS four-spacecraft
    burst mode data using the reciprocal vector method for spatial gradients.

    Pi-D = -Π:D  (incompressive pressure-strain interaction)
    p·θ  = p * ∇·u  (compressive pressure-strain interaction)

    Parameters
    ----------
    prestensor : array-like, shape (N, 3, 3)
        FPI pressure tensor level 2 product from one spacecraft. Units in nPa.

    positions : array-like, shape (4, N, 3)
        MEC spacecraft positions interpolated onto FPI time grid. Units in km.
        Axis 0 indexes spacecraft (0=MMS1, 1=MMS2, 2=MMS3, 3=MMS4).

    bulkv_arr : array-like, shape (4, N, 3)
        FPI bulk velocity from all four spacecraft interpolated onto FPI time
        grid. Units in km/s. Axis 0 indexes spacecraft.

    Returns
    -------
    pi_d : ndarray, shape (N,)
        Pi-D time series. Units in nPa/s.

    p_theta : ndarray, shape (N,)
        Pressure dilatation time series. Units in nPa/s.

    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid values.
    TypeError
        If inputs cannot be converted to numpy arrays.

    Notes
    -----
    All inputs must be in the same coordinate frame (GSE recommended).
    Positions must be interpolated onto the FPI burst mode time grid before
    calling this function (MEC cadence ~30s vs FPI burst ~150ms).
    The full pressure-strain interaction is p·θ + Pi-D.

    References
    ----------
    Yang et al. (2017) Phys. Rev. E 95, 061201
    Yang et al. (2022) ApJ 929, 142
    Cassak & Barbhuiya (2022) Phys. Plasmas 29, 122306
    '''

    # Input validation and conversion
    try:
        prestensor = np.asarray(prestensor, dtype=float)
        positions  = np.asarray(positions,  dtype=float)
        bulkv_arr  = np.asarray(bulkv_arr,  dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {e}")

    # Shape checks
    if prestensor.ndim != 3 or prestensor.shape[1:] != (3, 3):
        raise ValueError(
            f"prestensor must have shape (N, 3, 3), got {prestensor.shape}"
        )

    N = prestensor.shape[0]

    if positions.shape != (4, N, 3):
        raise ValueError(
            f"positions must have shape (4, N, 3), got {positions.shape}. "
            f"Expected N={N} to match prestensor. "
            f"Ensure positions are interpolated onto the FPI time grid."
        )
    if bulkv_arr.shape != (4, N, 3):
        raise ValueError(
            f"bulkv_arr must have shape (4, N, 3), got {bulkv_arr.shape}. "
            f"Expected N={N} to match prestensor."
        )

    # Check for non-finite values
    if np.any(~np.isfinite(prestensor)):
        raise ValueError("prestensor contains NaN or Inf values")
    if np.any(~np.isfinite(positions)):
        raise ValueError("positions contains NaN or Inf values")
    if np.any(~np.isfinite(bulkv_arr)):
        raise ValueError("bulkv_arr contains NaN or Inf values")

    # Reciprocal vectors from spacecraft positions
    r_bary = positions.mean(axis=0)                          # (N, 3)
    d      = positions - r_bary[np.newaxis, :, :]            # (4, N, 3)
    R      = np.einsum('aNi,aNj->Nij', d, d)                 # (N, 3, 3)
    R_inv  = np.linalg.inv(R)                                # (N, 3, 3)
    k      = np.einsum('Nij,aNj->Nai', R_inv, d)             # (N, 4, 3)

    # Velocity gradient tensor: (∇u)_ij = ∂u_i/∂x_j
    grad_u = np.einsum('aNi,Naj->Nij', bulkv_arr, k)         # (N, 3, 3)

    # Deviatoric pressure tensor: Π = P - (1/3)Tr(P) I
    p_scalar = np.trace(prestensor, axis1=1, axis2=2) / 3.0  # (N,)
    Pi = prestensor.copy()
    for i in range(3):
        Pi[:, i, i] -= p_scalar

    # Traceless strain-rate tensor: D = sym(∇u) - (1/3)(∇·u) I
    S     = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
    div_u = np.trace(grad_u, axis1=1, axis2=2)               # (N,)
    D = S.copy()
    for i in range(3):
        D[:, i, i] -= div_u / 3.0

    # Pi-D = -Π:D and pressure dilatation = p·θ
    pi_d    = -np.einsum('Nij,Nij->N', Pi, D)                # (N,)
    p_theta = p_scalar * div_u                                # (N,)

    return pi_d, p_theta
