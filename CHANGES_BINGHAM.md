# Changes: Bingham orientation & Hessian (2025-09-09)

- Fixed Bingham quaternion handling to be **WORLD-frame** consistent.
- Implemented full quaternion left/right multiply matrices and Jacobians.
- Added `energy_grad_hess` to compute projected gradient/Hessian on the unit quaternion manifold.
- Rewrote `MultiFrameWeirdEKF.update`:
  - Uses WORLD-frame quaternion for each link.
  - Chains gradient/Hessian via `dq/dω_world` and the frame's WORLD angular Jacobian.
  - Maps to x=log(kp) space via finite-difference sensitivity of the equilibrium solver.
  - Combines with prior using a Laplace-style EKF update.
- Kept ASCII-only variable names throughout, per user's coding style.

**Note:** `ObservationBuilder.build_A_multi` already constructs A from (WORLD gravity) vs (gravity-in-FRAME). No change needed there.