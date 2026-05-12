# Test Refactor Plan — `meas_extensions_scarlet`

**Date:** 2026-05-12
**Ticket:** DM-54841

---

## 1. Motivation

The DM-54841 audit (`audits/audit-2026-05-05.md`) records 52 open findings.
Many of them are subtle: peak-list misalignments (C-4), flag-inversion
(C-1), aggregate-update routed to the wrong record (C-5), per-band PSF
center drift (C-6), NaN poisoning in deconvolution (C-11), chi² division
by zero (DB-4), etc. Each fix risks silently changing the behavior of
the deblender on inputs that nobody is currently testing.

The existing `tests/test_deblend.py` is structured as a small number of
large, end-to-end tests built on top of a single `setUp()` fixture that
constructs an 8-source, 4-blend scene. That structure has served the
package well for regression smoke-testing, but for a ticket that will
touch nearly every file in the package, we want a denser, more
fine-grained safety net.

The user's framing: "building the test images is a time consuming
procedure and I don't want to have to do that too often" — empirically
the image construction itself is ~5 ms; what is expensive is running the
full pipeline (deconvolve + deblend ≈ 0.7 s for the current scene), and
what is *cognitively* expensive is unspooling the implications of a
single failing assertion in a 200-line test that exercises a dozen
different code paths.

**Goal:** restructure the test suite so that

- adding a test for a single behavior takes ~10 lines, not ~50,
- failing assertions point at one specific subsystem,
- expensive setup (image construction, deblender runs) is shared across
  the tests that consume it,
- coverage spans every public function with at least one round-trip or
  golden-result test (currently 86% line coverage, with significant gaps
  in foundational conversions — U-10 in the audit).

**Non-goals for this ticket:**

- Writing tests that demonstrate the audit's known bugs. Per the user's
  instructions, those tests are deferred and added alongside each fix.
  This document is updated as each fix lands so the regression test
  added in that commit is recorded here.
- Changing any production code. Pure test reorganization.
- Replacing `lsst.utils.tests.TestCase` with bare `pytest`. The LSST
  convention is `lsst.utils.tests.TestCase` + `MemoryTestCase`; we keep
  it.

---

## 2. Current state

### 2.1 File inventory

```
tests/
  SConscript
  data/v29_models.json
  test_deblend.py        # 831 lines — mixes deblend, utils, PSF, IO
  test_isolated_source.py # 167 lines — IsolatedSource only
  utils.py               # 114 lines — SersicModel, PsfModel, initData
```

### 2.2 Test class breakdown in `test_deblend.py`

| Class | Tests | Subjects |
|---|---|---|
| `TestDeblend` | 6 | full pipeline: deconvolve, deblend, persistence, legacy IO |
| `TestUtils` | 9 | `scarletBoxToBBox`/`bboxToScarletBox`, `computeNearestPsf`, `computeNearestPsfMultiBand`, `buildObservation` |
| `MemoryTester` | (file-descriptor leak check) | inherited from `lsst.utils.tests` |

### 2.3 Pain points

1. **Monolithic setup.** `TestDeblend.setUp` builds the same 8-source
   scene for every test. The full deblend runs inside each test
   (`initialize_data` → `deconvolve` → `deblendTask.run`). The work is
   redone six times per file.

2. **One scene for everything.** All deblend tests share one model
   layout. Behaviors that depend on specific layouts (e.g. a parent
   with no intersecting deconvolved footprint, C-13; pseudo-peaks
   interleaved with real peaks, C-4) cannot be probed without forcing
   that scene to grow even further.

3. **Two unrelated concerns in one file.** `TestUtils` lives in
   `test_deblend.py` because it shares some helpers but tests a
   completely different layer of the package (PSF utilities). The
   `IsolatedSource` tests live separately in `test_isolated_source.py`.

4. **No round-trip tests for foundational conversions** (U-10):
   `scarletBoxToBBox`↔`bboxToScarletBox`, `afwFootprintToScarlet`↔
   `scarletFootprintToAfw`, `to_data`↔`to_source`. A silent flip
   in any of these propagates everywhere.

5. **No direct tests for** `metrics.py` (blendedness, fluxOverlap),
   `footprint.py` (peak conversion — C-10 lives here), or the io
   migration registry (`_to_1_0_0`, `_to_1_0_1` — IO-3 lives here).

6. **`test_footprints` is a 165-line test method.** It loops over
   `useFlux ∈ {False, True} × band ∈ ("g","r","i")`, runs the deblender
   once, then makes about a dozen assertions about catalog structure,
   peak metadata, heavy-footprint contents, model reconstruction, flux
   redistribution, ID monotonicity, and parent/child counts. When it
   fails, the message says "test_footprints failed" — not "peak
   coordinate for child 7 in band r is wrong by 2 px".

---

## 3. Proposed structure

### 3.1 Directory layout

```
tests/
  SConscript                       # unchanged
  data/                            # unchanged (v29_models.json + future fixtures)
    v29_models.json
  conftest.py                      # NEW: pytest discovery + shared autouse hooks
  scenes.py                        # NEW: named, parametrized model scenes
  pipeline.py                      # NEW: helpers that run stages (deconvolve, deblend, ...)
  utils.py                         # kept; lightly extended

  # one file per source module
  test_box_conversions.py          # utils.scarletBoxToBBox/bboxToScarletBox round-trips
  test_footprint_conversions.py    # footprint.py: afwFootprintToScarlet, peak conventions
  test_psf_utilities.py            # utils.computeNearestPsf*, buildObservation, multiband_convolve
  test_metrics.py                  # metrics.setDeblenderMetrics
  test_source.py                   # source.IsolatedSource (subsumes test_isolated_source.py)
  test_deconvolve_task.py          # DeconvolveExposureTask
  test_deblend_task.py             # ScarletDeblendTask (catalog structure, flags, skips)
  test_io_source_data.py           # IsolatedSourceData round-trips, span_array, peak types
  test_io_model_data.py            # migrations, CURRENT_SCHEMA, LsstScarletModelData
  test_io_persistence.py           # butler put/get, legacy model loading
```

This is one test file per source module (plus three test-fixture-only
files: `conftest.py`, `scenes.py`, `pipeline.py`). Within each test file
we use `lsst.utils.tests.TestCase` for assertions and `setUpClass` for
sharing fixtures. We delete `test_isolated_source.py` in favor of
`test_source.py`.

### 3.2 `scenes.py` — atomic, named test scenes

A *scene* is a list of `DeblenderTestModel` instances plus the metadata
needed to build it (bands, image PSF, model PSF). Currently the test
file inlines one big scene with 8 sources spanning 4 blends. We
factor each blend into its own scene, plus a few targeted edge-case
scenes:

```python
# scenes.py — sketch

@dataclass(frozen=True)
class Scene:
    name: str
    bands: tuple[str, ...]
    models: list[DeblenderTestModel]   # constructed lazily via .build()
    description: str

SCENES = {
    "one_isolated_psf":      # isolated point source, away from any neighbor
    "one_isolated_sersic":   # isolated extended source
    "two_psf_blend":         # two PSFs separated by < detection-merge distance
    "psf_plus_sersic_blend": # PSF on the wing of an extended source
    "three_source_blend":    # one Sersic + two PSFs
    "large_two_sersic":      # two large Sersics overlapping
    "edge_source":           # source whose footprint touches the image edge
    "saturated_pixel":       # source with a SAT mask bit set at the peak
    "off_image_negative":    # source whose bbox dips below (0, 0)  [error path]
    "all_isolated":          # several PSFs all isolated, no blends
    "multi-blend":           # the current 8-source / 4-blend scene
}
```

Each scene is built by a factory function in `scenes.py`. Scenes are
constructed at most once per test session because `pipeline.py` caches
the resulting numpy arrays (see §3.4).

**Why factor by *blend*, not by *individual source*:** the deblender
operates per parent footprint. Per-blend scenes let us test parent-level
behavior with one parent per scene. Cross-blend interactions (e.g.
parent IDs are unique and monotone across blends) are tested with the
`multi-blend` scene.

### 3.3 `pipeline.py` — staged execution

The current `initialize_data` runs everything from image-building to
catalog construction in one shot. We split into named, individually
cacheable stages:

```python
def build_image(scene: Scene) -> ImageBundle:
    """Render deconvolved + convolved + noise + multiband exposure."""

def detect(image: ImageBundle, config=None) -> DetectionBundle:
    """Run SourceDetectionTask, return catalog + schema mapper."""

def deconvolve(image: ImageBundle, detection: DetectionBundle, config=None) -> DeconvolveBundle:
    """Run DeconvolveExposureTask in every band."""

def deblend(image, detection, deconvolved, config=None) -> DeblendBundle:
    """Run ScarletDeblendTask."""
```

Each function is **pure** (no `self`), returns a frozen dataclass
("Bundle"), and is memoized in a session-scoped cache keyed by
`(scene_name, frozen_config_tuple)`. Configs are converted to a
hashable representation by serializing `config.toDict()` once.

Tests that **mutate** their result (e.g. anything that runs
`updateCatalogFootprints` and writes into the catalog) call
`bundle.copy()` first.

Cost on the current `multi-blend` scene:
- `build_image`: ~5 ms (cheap; cache mainly for ergonomics)
- `detect`: ~30 ms
- `deconvolve`: ~80 ms
- `deblend`: ~600 ms

Caching shaves the deblend cost down to "once per (scene, config)
combination" instead of "once per test method".

### 3.4 Fixture caching strategy

Two layers of cache:

1. **Module-level `lru_cache`** on the four pipeline functions. Hashable
   key is `(scene_name, config_hash)`. Stored as private module state in
   `pipeline.py`.

2. **`setUpClass` per test class**, which fetches the bundle it needs
   from `pipeline.py` and binds it to `cls.bundle`. This keeps the test
   code looking like normal unittest.

The cache is process-local and dies with the pytest process. We do not
pickle bundles to disk — runtime is small enough that the disk cache
isn't worth the complexity, and re-running with `pytest --lf` benefits
from the in-process cache anyway.

**Concurrency note:** when the test suite is run under
`pytest-xdist -n N`, each worker has its own cache. That's fine — we
still amortize across tests within a worker.

### 3.5 `conftest.py`

- Calls `lsst.utils.tests.init()` once at collection time (replaces the
  per-file `setup_module`).
- Exposes a `--scene` option for ad-hoc developer use (`pytest -k "..."
  --scene=one_isolated_psf` selects a scene at runtime in tests that
  honor it).
- Registers no autouse fixtures that hide behavior — the only
  autouse-thing is the `lsst.utils.tests.init()` hook.

---

## 4. Test inventory — what we add, where it lives

Below, **(Existing)** means a test we lift from the current files,
possibly renamed or split; **(New)** means a test that does not exist
today and is added now to give the audit fixes a regression baseline.
Tests that demonstrate open audit findings are not listed here — they
ship with their respective fix commits and are added to this document
at that time.

### 4.1 `test_box_conversions.py`

- **(New)** `test_scarletBoxToBBox_roundtrip` — random origins,
  shapes; assert `bboxToScarletBox(scarletBoxToBBox(b)) == b`.
- **(New)** `test_bboxToScarletBox_roundtrip` — inverse direction.
- **(New)** `test_box_negative_origin` — origin below (0,0); both
  conversions agree on the signed offset.
- **(New)** `test_box_with_xy0_offset` — non-zero `xy0` parameter.
- **(Existing)** `test_box_transforms` — lifted as-is from
  `TestUtils.test_box_transforms` for backward continuity.

Audit links: U-10 (round-trip gap), U-15 (the `xy0` param meaning;
already discussed and discarded but the new tests pin it down).

### 4.2 `test_footprint_conversions.py`

- **(New)** `test_afwFootprintToScarlet_peak_order` — build an afw
  Footprint with known peak positions, assert the scarlet `Peak`
  objects have `peak.y == iy` and `peak.x == ix`.
- **(New)** `test_scarletFootprintToAfw_peak_order` — inverse.
- **(New)** `test_roundtrip_footprint_with_negative_origin` — bbox
  origin negative, peaks at integer positions.
- **(New)** `test_roundtrip_footprint_empty_spans` — degenerate case.
- **(New)** `test_roundtrip_footprint_edge_pixels` — spans on row 0,
  column 0.
- **(New)** `test_scarletFootprintsToPeakCatalog_schema` — build a
  PeakTable schema and verify columns.

Audit links: C-10 (discarded), U-10 (round-trip gap).

### 4.3 `test_psf_utilities.py`

- **(Existing)** `test_computeNearestPsfGood`,
  `test_computeNearestPsfRecoverable`, `test_computeNearestPsfBad`.
- **(Existing)** `test_computeNearestPsfMultiBandGood`,
  `..._Recoverable`, `..._Incomplete`, `..._Bad`.
- **(Existing)** `test_buildObservationBadPsfs`.
- **(New)** `test_multiband_convolve_2d_psf_broadcasts` — verify that
  a 2-D PSF passed in gets broadcast across bands correctly.
- **(New)** `test_multiband_convolve_per_band_psf` — different PSFs per
  band produce different convolved outputs.

Audit links: C-6, U-1; the rest pin existing behavior.

### 4.4 `test_metrics.py`

- **(New)** `test_setDeblenderMetrics_isolated` — single-source blend;
  `maxOverlap`, `fluxOverlap`, `fluxOverlapFraction`, `blendedness` all
  equal expected values (overlap = 0, blendedness = 0).
- **(New)** `test_setDeblenderMetrics_two_disjoint` — two
  non-overlapping sources; overlap metrics are zero.
- **(New)** `test_setDeblenderMetrics_overlapping_psfs` — two PSFs
  with overlap; assert metric values against an analytic expectation
  (or freeze the current value as a regression baseline).

Audit links: U-3, U-8 (type hint — not a runtime test),
U-11 (algorithmic improvement — not a runtime test).

### 4.5 `test_source.py` (replaces `test_isolated_source.py`)

- **(Existing)** all current tests from `test_isolated_source.py`:
  `test_constructor`, `test_copy`, `test_deep_copy`, `test_slice`,
  `test_reorder`, `test_subset`, `test_indexing_errors`.
- **(New)** `test_from_footprint_roundtrip` — build a Footprint,
  call `IsolatedSource.from_footprint`, assert peak and bbox match.
- **(New)** `test_to_data_from_data_roundtrip` — `to_data` →
  `to_source` (via `IsolatedSourceData.to_source`) preserves model
  data, peak, origin.

Audit links: U-4, U-9, U-12, U-13, U-14 (cleanup-only).

### 4.6 `test_deconvolve_task.py`

- **(Existing)** `test_default_deconvolve` — pulled and slimmed: just
  asserts that the deconvolved image matches the truth model within
  noise tolerance.
- **(Existing)** `test_catalog_free_deconvolve` — same, with
  `useFootprints=False`.
- **(New)** `test_deconvolve_one_isolated_psf` — minimal scene, exact
  pixel-level expectation.
- **(New)** `test_deconvolve_preserves_image_metadata` — output
  exposure has the right PSF, WCS, dimensions.
- **(New)** `test_deconvolve_with_nan_input` — input image with a
  NaN pixel; assert the deconvolver does not propagate NaN to the
  output. (Note: this exercises the area C-11 lives in; if the
  audit's C-11 fix changes the contract, this test moves with it.)

Audit links: C-11, C-12, DC-1, DC-3, DC-4, DC-7, DC-8.

### 4.7 `test_deblend_task.py`

This is the heart of the refactor. The current monolithic
`test_footprints` is split into many small tests, each scoped to one
behavior. They share a single class-level cached deblend run on the
`multi-blend` scene, plus per-scene cached runs for targeted tests.

Group 1 — *Catalog structure* (uses `multi-blend`, cached):

- **(Existing/New)** `test_catalog_total_count` — n rows in output
  equals n input models.
- **(New)** `test_isolated_parents_marked` — the `deblend_skipped_isolatedParent`
  flag is set exactly on isolated single-peak parents.
- **(Existing)** `test_isolated_source_persisted` — `modelData.isolated`
  contains a `SourceData` for each isolated parent; spans match.
- **(New)** `test_child_ids_above_parent_ids` — every child id is
  greater than `max(parent_ids)`.
- **(New)** `test_catalog_sorted_by_parent_id` — assertion lifted from
  current `test_footprints`.
- **(New)** `test_every_source_has_one_peak` — assertion lifted.
- **(New)** `test_nChild_consistency` — sum of `deblend_nChild` over
  parents equals number of children.

Group 2 — *Heavy footprints* (uses `multi-blend`, cached; one method
per assertion currently bundled in `test_footprints`):

- **(New)** `test_heavy_footprint_flux_at_peak` — `deblend_peak_instFlux`
  equals the model flux at the peak coordinate. Parametrized over band
  and `useFlux ∈ {True, False}`.
- **(New)** `test_heavy_footprint_peak_position` — peak in the
  HeavyFootprint matches `deblend_peak_center_{x,y}`.
- **(New)** `test_heavy_footprint_matches_model` — pixel-level
  comparison of the HeavyFootprint to the scarlet model.
  Parametrized over band and `useFlux`.

Group 3 — *Skip / failure semantics* (uses per-scene runs):

- **(Existing)** `test_skip_too_big` — `maxFootprintArea=2000` causes
  the large_two_sersic blend to skip with `deblend_skipped_parentTooBig`.
- **(Existing)** `test_skip_too_many_peaks` — `maxNumberOfPeaks=2`
  causes the three_source_blend to skip with `deblend_skipped_tooManyPeaks`.
- **(New)** `test_skip_doesnt_affect_other_parents` — one parent is
  skipped, others succeed normally.

Group 4 — *Pseudo-source filtering* (uses a new scene with a
pseudo-source):

- **(New)** `test_pseudo_source_excluded_from_children` — pseudo-peaks
  do not produce deblend children.

Audit links for this file: C-1, C-4, C-5, C-13, DB-4, DB-5, DB-10,
DB-11, DB-17, plus the cleanup items DB-2, DB-3, DB-7, DB-8, DB-9,
DB-12, DB-13, DB-18, DB-19, DB-20, DB-21.

### 4.8 `test_io_source_data.py`

- **(New)** `test_isolated_source_data_roundtrip` — build a
  `IsolatedSourceData`, serialize to dict, parse back, assert
  equality of all fields.
- **(New)** `test_span_array_roundtrip` — round-trip a non-trivial
  span_array; assert bit-exact recovery.

Audit links: IO-1, IO-4, IO-7.

### 4.9 `test_io_model_data.py`

- **(New)** `test_to_1_0_0_adds_isolated_key` — pre-1.0.0 data has
  no `isolated` key; migration adds it as `{}`.
- **(New)** `test_to_1_0_1_adds_footprint_metadata` — pre-1.0.1 data
  has no `metadata.footprint`; migration adds it.
- **(New)** `test_schema_version_constants_match` — assert
  `SCARLET_LITE_SCHEMA` and `CURRENT_SCHEMA` are wired correctly.

Audit links: C-7, IO-3.

### 4.10 `test_io_persistence.py`

- **(Existing)** `test_persistence` — split into:
  - `test_butler_put_get_roundtrip` — core round-trip.
  - `test_butler_get_single_blend_parameter` — `parameters={"blend_id": ...}`.
  - `test_butler_get_multiple_blend_parameter` — `parameters={"blend_id": [...]}`.
- **(Existing)** `test_legacy_model` (v1.0.0 LsstScarletModelData
  storage class).
- **(Existing)** `test_older_legacy_model` (v0 ScarletModelData
  storage class with override).

Audit links: C-3, IO-5, IO-6.

---

## 5. Migration plan

Stepwise, so that the test suite is green at every step. Each step is
its own commit.

1. **Add `scenes.py`, `pipeline.py`, `conftest.py`**. Do not touch
   existing test files. Confirm `pytest tests/` is still green.

2. **Add the new round-trip tests** (`test_box_conversions.py`,
   `test_footprint_conversions.py`, `test_metrics.py`,
   `test_io_source_data.py`, `test_io_model_data.py`). These are
   purely additive — no risk to existing tests.

3. **Split `test_deblend.py::TestUtils`** out into
   `test_psf_utilities.py`. Lift `TestUtils._generate*` helpers either
   into `pipeline.py` or keep them local to the new file. Delete
   the moved class from `test_deblend.py`.

4. **Split `test_deblend.py::TestDeblend::test_persistence`,
   `test_legacy_model`, `test_older_legacy_model`** into
   `test_io_persistence.py`. Move `_setup_butler` and `_test_blend`
   helpers along with them.

5. **Split `test_default_deconvolve` and `test_catalog_free_deconvolve`**
   into `test_deconvolve_task.py`. Both already self-contained.

6. **Split `test_skipped`** into `test_deblend_task.py` (Group 3).

7. **Decompose `test_footprints`** — the biggest piece. One commit
   per assertion-group: catalog structure (Group 1), heavy footprints
   (Group 2), each as its own test method on a class that caches the
   deblend run via `setUpClass`.

8. **Replace `test_isolated_source.py` with `test_source.py`** —
   includes everything currently there plus the new from_footprint
   and to_data round-trips.

9. **Delete the now-empty `TestDeblend` class** from `test_deblend.py`
   and rename the file to `test_deblend_task.py` (or merge into it
   if Step 6/7 already created it). At this point `test_deblend.py`
   no longer exists.

10. **Add `MemoryTester(lsst.utils.tests.MemoryTestCase)`** to one of
    the new files (probably `test_deblend_task.py`) so the
    file-descriptor leak check still runs.

11. **Verify coverage** with `pytest --cov` — should be ≥ current 86%
    (and ideally higher, given the new round-trip tests).

After step 11, the audit-fix work begins. Each fix commits its
deferred test alongside its code change.

---

## 6. Open questions for review

1. **Scope of `scenes.py`.** Do you want me to include a "real-world"
   scene seeded from a small cutout of an actual coadd, or keep
   everything synthetic? Synthetic is simpler and deterministic;
   real-world catches issues you can't predict.

A: No, we have CI for that

2. **`pytest-xdist` compatibility.** With multiple workers, each builds
   its own cache. Is that acceptable, or do you want a `pickle`-backed
   on-disk cache shared between workers? (My recommendation: leave as
   in-process only — disk caching adds complexity for a small win at
   ~6 s total test time.)

A: Agreed, per process cache is fine

3. **Per-class deblend runs vs single session-scope deblend.** I
   propose `setUpClass`-level caching: one deblend per test class per
   scene. An alternative is a single session-scoped fixture, but that
   makes mutation hazards harder to reason about. Worth deciding now.

A: Agreed, this is a clean separation.

4. **Should the `Bug — deferred` tests live as `@unittest.expectedFailure`
   today**, so we get a visible reminder for each finding, or are they
   strictly "deferred to that fix's commit"? My read of your
   instructions is the latter; flagging in case you want the former.

A: Don't add the deferred tests at all yet. There are some issues that might have slightly different fixes or behavior than your proposals, so I don't want to assume anything about them yet. We can update this doc as those issues are closed and the tests created.

5. **`utils.py`** currently lives at the top level of `tests/`.
   `scenes.py` and `pipeline.py` will sit next to it. Worth promoting
   to a `tests/_support/` package, or is flat fine? Flat is fine for
   now (LSST convention).

A: Yes, please keep it flat for now.

6. **Renaming.** `test_deblend.py` → `test_deblend_task.py` makes the
   one-file-per-source-module mapping clean, but Git will see this as
   a delete-add. Acceptable?

A: Agreed.