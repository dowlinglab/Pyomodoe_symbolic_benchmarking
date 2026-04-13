# Development Note: Alexandrian 2014 1D Toy (Pyomo.DAE + Pyomo.DoE, Central FD)

Date: 2026-04-13  
Workspace: `/Users/snarasi2/projects/Pyomo_DoE_CRC`

This note documents the 1D, 2-parameter “toy” workflow we built to mirror the Alexandrian et al. (2014) PDE inversion / MBDoE ideas in a frequentist/FIM setting, using `pyomo.dae` + `pyomo.contrib.doe` with **central finite-difference sensitivities** (`fd_formula="central"`).

The goal here is not Bayesian inference, but designing measurements to estimate a low-dimensional parameterization of the unknown initial condition.

## Model

PDE (1D advection-diffusion):

- `u_t = kappa*u_xx - v*u_x`, for `x in [0,1]`, `t in [0,1]`
- Neumann boundary conditions: `u_x(t,0)=0`, `u_x(t,1)=0`

Unknowns:

- A 2-parameter representation of the initial condition `m(x)`:
  - Patch basis (discontinuous at 0.5): `u(0,x)=theta1` if `x<0.5` else `theta2`
  - We intentionally started here because it makes “sensor placement matters” obvious: sensors placed only on one side can be weakly sensitive to one parameter.

Discretization:

- `pyomo.dae` `dae.finite_difference` in both `x` and `t`
- Typical grid used in our sweeps:
  - `nfe_x = 20` so `dx = 1/20 = 0.05`
  - `nfe_t = 40` so `dt = 1/40 = 0.025`
- We emphasized *grid-aligned* candidate `x` and `t` values (e.g. `x2=0.55,0.60,...`) to avoid indexing surprises.

## Making the Discretized PDE “Square”

When discretizing in Pyomo.DAE, second-derivative boundary points can leave degrees of freedom unless you provide enough BC/IC constraints.

For the Neumann BC case, we used:

- BCs: `u_x(t,0)=0`, `u_x(t,1)=0` for all `t`
- Dummy constraints at boundaries for second derivative and initial time derivative to eliminate leftover DoFs:
  - `u_xx(t,0)=0`, `u_xx(t,1)=0`
  - `u_t(0,x)=0`

This was validated with IDAES diagnostics (`degrees_of_freedom` and structural diagnostics).

## DoE Setup (Frequentist/FIM)

We used `pyomo.contrib.doe.DesignOfExperiments` with:

- `fd_formula="central"` and `step=1e-3` for sensitivities
- Cholesky FIM build (`_Cholesky_option=True`, `_only_compute_fim_lower=True`)
- Objective options used:
  - `"determinant"` for D-opt (maximize `log10 det(FIM)`)
  - `"trace"` for Pyomo’s A-like objective (maximize `log10 trace(FIM)`)
  - (Also computed E-opt and cond metrics from the computed FIM as diagnostics.)

Key labels on the experiment model:

- `model.unknown_parameters` (Suffix): `{theta1, theta2}`
- `model.experiment_outputs` (Suffix): the measured outputs (`Expression`s)
- `model.measurement_error` (Suffix): per-output measurement std dev (we used `1e-2`)
- `model.experiment_inputs` (Suffix): design variables (e.g. weights), or a dummy var when we are searching `x2`/`t2` externally in Python.

## “Sensor weights” (continuous relaxations)

We modeled optional/active sensors using weights `w in [0,1]` multiplying each output:

- `y = w * u(t_i, x_j)`

Important: without a budget/penalty, the optimal solution is trivial (`w -> 1` for all).

We explored selection/budget behavior with a 1D budget parameterization, e.g.:

- `w_right = 1 - w_left`

This collapses a 2D selection into a 1D sweep along a feasible line.

## Stage 1: Space Search (x2 sweep)

Goal: keep one sensor fixed (e.g. `x1=0.2`) and search the second sensor `x2>0.5` to improve:

- E-opt: maximize `lambda_min(FIM)` (weakest direction)
- cond: minimize `lambda_max(FIM)/lambda_min(FIM)` (conditioning)

Script:

- `/Users/snarasi2/projects/Pyomo_DoE_CRC/elliptic_PDE_Alexandrian_test_1D_Stage_1_Heatmap_theta2_improve_log10.py`

What it does:

- Sweeps `x2` on grid points `0.55, 0.60, ..., 0.95`
- For each `x2`, computes the FIM and records:
  - `log10 det(FIM)` (D-opt diagnostic)
  - `log10 trace(FIM)` (A-like diagnostic)
  - `log10 lambda_min(FIM)` (E-opt diagnostic)
  - `cond(FIM)`
- Saves curves (PNG+EPS) with the best point starred.

Observed behavior:

- E-opt plateaued for `x2>=0.9` under fixed times `{0.2, 0.5}`.

## Stage 2: Time Search (t2 sweep, then t1 sweep)

We separated “time weights” (reweighting fixed times) from “time search” (moving sampling instants).

The key improvement came from *searching sampling times on-grid*.

### t2 sweep (fix t1, sweep t2)

Script:

- `/Users/snarasi2/projects/Pyomo_DoE_CRC/elliptic_PDE_Alexandrian_test_1D_Stage_2_t2_search_log10.py`

What it does:

- Fixes sensors (defaults `x1=0.2`, `x2=0.9`)
- Fixes `t1=0.2` (editable)
- Sweeps `t2` over all interior grid points `dt, 2dt, ..., 1-dt` excluding `t2=t1`
- Computes FIM metrics and saves curves (PNG+EPS).

### t1 sweep (fix t2, sweep t1)

Script:

- `/Users/snarasi2/projects/Pyomo_DoE_CRC/elliptic_PDE_Alexandrian_test_1D_Stage_2_t1_search_log10.py`

What it does:

- Fixes sensors (defaults `x1=0.2`, `x2=0.9`)
- Fixes `t2=0.1` (editable)
- Sweeps `t1` over grid points from `0.1` to `1-dt`, excluding `t1=t2`
- Computes FIM metrics and saves curves (PNG+EPS).

### Choosing a final pair of times (example)

Based on the cond criterion, we ended up with a pair of sampling times:

- `{0.1, 0.125}`

and validated that the conditioning improved substantially (see next section).

## Validation Run: Conditioning Improvement

For one validation run with times `{0.1, 0.125}`, a representative FIM was:

- `F = [[79.92, 9.03], [9.03, 9.19]]`
- eigenvalues: `lambda_max≈81.05`, `lambda_min≈8.06`
- condition number: `cond(F)≈10.06`

Earlier baseline-ish setups had `cond(F)≈25`, so this is about a **2.5x improvement** in conditioning, meaning the parameter information is significantly less anisotropic.

## Criteria Notes (geometry intuition)

For a 2-parameter problem:

- D-opt: maximize `det(F)` (or `log det(F)`)
  - maximizes “volume” of information; can still be ill-conditioned if one eigenvalue is huge and the other modest.
- E-opt: maximize `lambda_min(F)`
  - directly lifts the weakest direction; good for identifiability balance.
- Conditioning: minimize `cond(F)=lambda_max/lambda_min`
  - promotes balanced eigenvalues; often improves numerical stability of estimation.
- Pyomo “trace” objective: maximizes `trace(F)` (and we plotted `log10 trace(F)`)
  - can favor piling information into already-strong directions; can disagree with E-opt/cond.

## Practical Implementation Notes

1. Indexing: state is `u[t, x]` (time first). Mis-ordering (`u[x,t]`) silently breaks intent.
2. Grid alignment: pick candidate `x` and `t` that are exactly on the discretization grid.
3. DoE requirements: `experiment_inputs` must exist. When searching `x2` or `t2` externally, we used a fixed `dummy_input` variable just to satisfy the interface.
4. `kaug`: not used in these runs. With `fd_formula="central"` we are doing central finite-difference sensitivities.
5. Results reporting: in some dumps, `results["Measurement Error"]` appeared identical to outputs; verify the suffix directly on the built model if needed.

## Key Files (Code)

- Final DoE run model:
  - `/Users/snarasi2/projects/Pyomo_DoE_CRC/elliptic_PDE_Alexandrian_test_1D_doe.py`
- Stage 1 space search:
  - `/Users/snarasi2/projects/Pyomo_DoE_CRC/elliptic_PDE_Alexandrian_test_1D_Stage_1_Heatmap_theta2_improve_log10.py`
- Stage 2 time searches:
  - `/Users/snarasi2/projects/Pyomo_DoE_CRC/elliptic_PDE_Alexandrian_test_1D_Stage_2_t2_search_log10.py`
  - `/Users/snarasi2/projects/Pyomo_DoE_CRC/elliptic_PDE_Alexandrian_test_1D_Stage_2_t1_search_log10.py`

