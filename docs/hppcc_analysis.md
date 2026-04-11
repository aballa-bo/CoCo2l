# HPPCC Document Analysis

## Documents

- `HPPCC_2005.pdf`
  - Original HPPCC method by Andersen and Hardeberg.
  - Uses a set of local `3x3` matrices.
  - Each matrix preserves:
    - the neutral patch / white point
    - two adjacent chromatic training patches
  - Regions are defined by hue angle in camera `rg` chromaticity.
  - Main strength: exact hue-plane preservation and exact fit of selected chromatic anchors.
  - Main weakness: only `C0` continuity and limited optimization freedom.

- `ieee_hppcc_wcm_final050716_accept_final.pdf`
  - Weighted Constrained Matrixing, referred to as `HPPCC-WCM`.
  - Trains one constrained `3x3` matrix per chromatic training patch.
  - For a new sample, computes a hue angle and uses the normalized weighted sum of all matrices.
  - Weight is a power function of hue-angle distance; exponent `p` is optimized globally.
  - Main strength: smooth (`C∞` except at the neutral axis) and still hue-plane preserving.
  - Main weakness: more expensive at inference, more complex training, and the weighting exponent must be tuned.

- `josaa-33-11-2166.pdf`
  - New HPPCC method based on constrained least squares; effectively a region-wise improved HPPCC.
  - Sorts all samples by hue angle, partitions them into `K` hue regions, and solves all regional matrices jointly.
  - Adds explicit constraints for:
    - continuity at region boundaries
    - white-point preservation
  - Closed-form solution is given through a constrained least-squares system.
  - Main strength: robust, uses all samples, preserves hue planes, remains exposure/shading invariant, and improves over the original HPPCC.
  - Main weakness: still piecewise linear and only `C0` continuous, unless moved to the weighted method.

## Key Geometry Shared by the Papers

- Work in **linear camera RGB**. This is essential.
- Convert camera RGB to `rg` chromaticity:
  - `r = R / (R + G + B)`
  - `g = G / (R + G + B)`
- White-balanced neutral maps to `(1/3, 1/3)` in `rg`.
- Hue angle is computed from the offset from the neutral point.
- A hue plane is the plane spanned by:
  - the neutral axis
  - one chromatic camera color
- Linear transforms preserve hue planes; general nonlinear polynomial methods do not.

## Best First Implementation Choice

The best starting point for this project is the **2016 JOSA method** from `josaa-33-11-2166.pdf`.

Reason:

- It is the most implementation-friendly formulation.
- It uses **all training patches**, not just boundary patches.
- It gives a **closed-form constrained least-squares solution**.
- It matches the desired application flow well:
  - measure ColorChecker patches from a RAW image
  - compare to expected XYZ values
  - optimize correction quality while preserving hue planes

The 2005 paper is still important because it defines the original geometric idea and the `rg` / hue-region construction. The 2015 weighted paper is a good second-stage upgrade if we later want smoother transitions between hue regions.

## Recommended App Pipeline

### 1. RAW decoding

Use `libraw` through Python bindings. The papers assume **linear sensor data**, not gamma-corrected rendered RGB.

Important requirement:

- We must keep access to **linear, demosaiced camera RGB**.
- We should disable or avoid:
  - automatic tone curves
  - auto-brightening
  - output color transforms
  - gamma encoding

### 2. Camera RGB preparation

Before any color correction:

- subtract black level
- normalize by white level / saturation level
- demosaic
- white-balance using RAW metadata or a chosen neutral reference

The papers distinguish:

- white balancing: preprocessing
- white-point preservation: enforced property of the correction model

These are not the same thing.

### 3. Patch extraction

Detect or manually define the 24 X-Rite ColorChecker Classic patches.

For each patch:

- compute a robust average camera RGB value from a central patch region
- avoid borders, shadows, specular highlights, and clipping

Even though the user described transforming the whole image to XYZ before patch comparison, for model fitting we do **not** need an initial per-pixel camera-to-XYZ transform first. The papers learn the correction **from measured camera RGB patch values directly to reference XYZ patch values**.

### 4. Reference XYZ values

Use expected XYZ values for the exact ColorChecker target and illuminant/observer combination.

This is critical:

- the reference XYZ values must match the illuminant under which the chart is imaged, or be adapted consistently
- if the light is not standardized, measured patch XYZ values from a spectro/colorimeter are better than vendor nominal values

### 5. Train the correction model

#### Option A: baseline

Fit a single white-point-preserving `3x3` matrix.

This gives:

- a sanity-check baseline
- exposure invariance
- a simpler benchmark before HPPCC

#### Option B: first HPPCC implementation

Implement the 2016 constrained least-squares formulation:

- sort the training patches by hue angle
- partition them into `K` hue regions
- assemble block-diagonal matrix `A`
- assemble continuity and white-point constraints in matrix `C`
- solve:

`min ||A T - X|| subject to C T = B`

The paper gives the closed-form solution:

`[T; Z] = [[2 A^T A, C^T], [C, 0]]^-1 [2 A^T X; B]`

where:

- `T` contains the regional `3x3` matrices
- `Z` are Lagrange multipliers

### 6. Apply correction

For each pixel:

- compute hue angle in camera `rg`
- select the hue region
- apply the corresponding `3x3` matrix

This yields corrected XYZ.

### 7. Evaluate error

Convert corrected XYZ and reference XYZ to CIELAB and compute `ΔE00`.

Important detail:

- the papers often optimize in XYZ or CIELUV / CIELAB-like spaces, but your target is explicitly **minimize `ΔE00`**
- that means we should use `ΔE00` as the optimization score for:
  - choosing number of hue regions `K`
  - choosing hue-region boundaries
  - possibly choosing the neutral patch or white-balance strategy

## Important Divergence From the Papers

The papers do **not** optimize the model directly for `ΔE00`.

They mostly:

- solve matrix parameters by least squares in XYZ with constraints
- then report perceptual errors such as `ΔE`

For this project, a practical approach is:

1. solve the constrained least-squares matrices exactly as in the 2016 paper
2. use `ΔE00` as the outer objective for selecting model hyperparameters

This is the safest first version because direct `ΔE00` optimization over matrix coefficients would make the training nonlinear and much harder.

## Suggested Hyperparameters

For a first implementation:

- use the 18 chromatic patches for hue geometry
- use one neutral patch for white preservation
- test `K = 4` and `K = 6`

This recommendation comes from the 2016 paper, where performance gains usually plateau around that range.

## Practical Notes for RAW Work

- Patch measurements must come from **linear** data.
- Saturated or clipped patches must be rejected.
- The neutral patch used for white preservation should be stable and not near black.
- If the illuminant is unknown, nominal ColorChecker XYZ values may dominate the residual error more than the correction model itself.
- If patch detection is automatic, add a manual override path; patch localization errors will overwhelm model differences.

## Recommended Implementation Order

1. RAW loader returning linear demosaiced camera RGB.
2. ColorChecker patch sampler.
3. Reference patch dataset loader.
4. White-point-preserving single `3x3` baseline.
5. 2016 constrained HPPCC region-wise solver.
6. `ΔE00` evaluation and reporting.
7. Region-count / boundary optimization.
8. Optional 2015 weighted HPPCC-WCM smoother as a second-stage enhancement.

## Immediate Next Decision

For the first working version, implement:

- baseline white-point-preserving `3x3`
- 2016 constrained HPPCC with fixed equal-count hue regions
- `ΔE00`-based model selection across a small set of `K` values

That is the shortest path to a correct, testable application aligned with the documents.
