.. _lsst.meas.extensions.scarlet-changes:

############
Change guide
############

This page records user-facing changes to ``lsst.meas.extensions.scarlet``:
feature changes and major bug fixes that affect the values, flags, schema,
public API, persisted formats, or default behavior that users and downstream
tasks observe. Purely internal fixes (docstring corrections, dead-code
removal, logging tweaks, behavior-preserving refactors) are not listed here.

.. _lsst.meas.extensions.scarlet-changes-DM-54841:

DM-54841
========

Bug fixes to the deblender.

.. note::

   This ticket changes the meaning of values written into existing catalog
   fields. Pipelines or analysis code that read these fields should review
   the entries below before upgrading.

``deblend_blendConvergenceFailedFlag`` semantics corrected
----------------------------------------------------------

The parent-catalog flag ``deblend_blendConvergenceFailedFlag`` was previously
stored with **inverted** semantics: a blend that *converged* was flagged as
having failed, and a blend that did *not* converge was flagged as a success.
The flag now matches its documented meaning — it is set (``True``) when the
blend failed to reach convergence, and unset (``False``) when it converged.

.. warning::

   The boolean value of this column has flipped. Any analysis or selection
   that read ``deblend_blendConvergenceFailedFlag`` from catalogs produced by
   an earlier version was effectively selecting the opposite population.
   Code that worked around the bug by inverting the flag must drop that
   inversion. The flag is now set (``True``) only for a blend that was fit
   but did not converge within ``maxIter`` iterations. Successfully converged
   blends report ``False``, and so do parents where no fit was attempted
   (isolated and skipped parents) — for those the flag no longer carries a
   spurious value that could contaminate convergence statistics.

Deblended-source peak metadata corrected for blends with sky peaks
------------------------------------------------------------------

When a parent footprint contained a pseudo peak (a sky object, or any peak
flagged by a ``pseudoColumns`` field) ordered before a real peak, each
deblended child was assigned the wrong detection ``PeakRecord``. The
affected children carried the wrong ``deblend_peakId``, the wrong
``deblend_peak_center_x`` / ``deblend_peak_center_y``, and the wrong values
for any peak-schema columns copied onto the child. The peak each child is
matched to is now taken from the pseudo-filtered peak list, so this metadata
is correct.

.. warning::

   Catalogs produced by an earlier version have shifted peak metadata for
   the children of any blend whose footprint included a pseudo peak. The
   source models themselves are unaffected — only the per-child peak columns
   listed above. Re-run the deblender to obtain corrected values.

All-sub-blends-failed summary now lands on the parent record
------------------------------------------------------------

When every sub-blend of a multi-peak parent failed or was skipped, the
aggregate ``deblend_*`` summary (``deblend_nPeaks``, ``deblend_nChild``,
``deblend_iterations``, ``deblend_chi2``, ``deblend_runtime``,
``deblend_spectrumInitFlag``, ``deblend_blendConvergenceFailedFlag``) was
written to the *last* sub-blend's catalog record instead of the parent's.
The parent record was left at schema defaults and the trailing sub-blend's
own per-sub-blend values were overwritten. The summary now lands on the
parent record as intended.

.. warning::

   Catalogs produced by an earlier version have schema-default values for
   the aggregate ``deblend_*`` columns on any parent whose sub-blends all
   failed, and the trailing sub-blend of such a parent carries clobbered
   per-sub-blend values. Re-run the deblender to obtain the correct
   parent-level summary and intact per-sub-blend records.

Per-source ``deblend_chi2`` no longer leaks neighbor residuals
--------------------------------------------------------------

The per-child ``deblend_chi2`` reduced-chi2 value is built from the
blend-level chi2 image, which carries residuals wherever the *combined*
blend model is positive. Within one child's bbox that footprint included
pixels that belong to a neighboring source's model, so the chi2 sum
attributed to the child also included the neighbor's residual at those
pixels. The sum was then normalized by just this child's positive-pixel
area, inflating the reported reduced chi2 whenever a neighbor overlapped
the child's bbox.

The chi2 sum is now masked by the child's own positive-model footprint
before normalization, so the numerator and denominator are over the same
pixel set. Children with no overlapping neighbor are unaffected;
children in overlapping blends now report a lower, more representative
``deblend_chi2``.

.. warning::

   Catalogs produced by an earlier version overstate ``deblend_chi2``
   for any deblended child whose bbox overlaps a neighboring source's
   positive-model footprint — i.e. essentially every child of a
   multi-peak blend. Selections or thresholds on ``deblend_chi2`` may
   need to be retuned, and any earlier comparison of values from
   isolated vs. blended sources was systematically biased.

Multiband PSF kernel image is sampled at a single sky position when possible
----------------------------------------------------------------------------

``computeNearestPsfMultiBand`` previously carried each band's fallback
location into the next band's search, so each band's PSF could end up
sampled at a different sky point. Those per-band PSFs were then projected
onto a single union bounding box and stacked as if they were co-centered,
producing a multiband PSF kernel image with hidden per-band centroid
offsets — a source of small but systematic astrometric biases in the
deblended catalog whenever the PSF model required a fallback location.

The search now tries the requested center in every band first; only bands
that fail at the center participate in a fallback search, which walks
catalog peak positions sorted by distance from the *requested* center and
accepts the first position that works in every failing band. When that
position is also valid for the bands that succeeded at the center, all
bands switch to it, so the multiband PSF is genuinely sampled at one sky
position. When it is not, the successful bands keep their center PSF and
only the failing bands use the common fallback; this two-location result
is logged at ``WARNING``. When no single fallback location works for
every failing band, each failing band uses its own nearest valid position
and the per-band mis-registration is also logged at ``WARNING``.

.. warning::

   Catalogs produced by an earlier version may carry small per-band
   astrometric offsets in the PSF kernel image used for objects whose PSF
   model was invalid at their detected center. Re-run the deblender to
   obtain coherent multiband PSFs at affected positions.

``merge_peak_*`` flags now propagate to deconvolved sub-blend parents
---------------------------------------------------------------------

When a parent footprint was subdivided into deconvolved sub-blends, the
extra peak-schema fields (``merge_peak_sky`` and any other
``merge_peak_*`` band/pseudo flags supplied via ``peakSchema``) were
silently dropped on the deconvolved sub-blend parent records: the
peak-to-parent schema mapper had only the peak minimal schema and no
mappings for the extra peak fields, so the values from the first peak
never made it onto the parent row. These flags are now copied onto the
deconvolved sub-blend parent records the same way they have always been
copied onto top-level parent and child records.

.. warning::

   Catalogs produced by an earlier version have every ``merge_peak_*``
   column equal to ``False`` on rows where ``parent`` is a deconvolved
   sub-blend parent (i.e. an entry in the ``objectParents`` table whose
   ``parent`` field is nonzero), regardless of the underlying peak flags.
   Selections that filtered these rows by ``merge_peak_*`` were silently
   dropping every row.

Per-band blend reconstruction now takes a band name
---------------------------------------------------

``lsst.meas.extensions.scarlet.io.updateBlendRecords`` now takes the
full ``modelData: LsstScarletModelData`` plus a band name
(``band: str``) and delegates per-child blend reconstruction to
``ScarletBlendData.minimal_data_to_blend(...)[band]`` from
``lsst.scarlet.lite``. ``buildMonochromaticObservation`` likewise gains
a required ``band: str`` argument and tags the observation with the
real band label instead of the previous ``"dummy"`` placeholder. The
top-level ``updateCatalogFootprints`` signature is unchanged.

The new flow preserves the original component types (factorized vs.
cube) under the per-band slice, instead of always collapsing to
``CubeComponent`` the way the previous hand-rolled reconstruction did.

.. warning::

   Callers that drove per-band ``Blend`` reconstruction directly
   through ``updateBlendRecords`` must switch their ``bandIndex: int``
   argument to ``band: str`` and additionally supply the
   ``modelData``. Code that built observations via
   ``buildMonochromaticObservation`` must also pass ``band``; the
   ``("dummy",)`` band labeling convention is gone.

``monochromaticDataToScarlet`` is deprecated
--------------------------------------------

``lsst.meas.extensions.scarlet.io.monochromaticDataToScarlet`` now
emits a ``FutureWarning`` on every call and will be removed after
v31. The function predates scarlet_lite's
``ScarletBlendData.minimal_data_to_blend`` and ``Blend.__getitem__``
slicing; that pair expresses the same job in three lines with a
better-behaved component round-trip. Migrate to::

   blend = blendData.minimal_data_to_blend(
       model_psf=modelData.metadata["model_psf"][None, :, :],
       psf=modelData.metadata["psf"],
       bands=modelData.metadata["bands"],
   )[band]

``monochromaticBand`` and ``monochromaticBands`` are deprecated
---------------------------------------------------------------

The module-level constants
``lsst.meas.extensions.scarlet.io.utils.monochromaticBand`` (``"dummy"``)
and ``monochromaticBands`` (``("dummy",)``) now emit a
``FutureWarning`` on access via a module ``__getattr__`` hook. They
will be removed after v31 alongside ``monochromaticDataToScarlet``.
Use the real band name from ``modelData.metadata["bands"]`` instead of
the ``"dummy"`` placeholder.

``loadBlend`` accepts ``modelData``; ``model_psf`` is deprecated
----------------------------------------------------------------

The public ``lsst.meas.extensions.scarlet.io.loadBlend`` helper now
accepts an optional ``modelData: LsstScarletModelData`` keyword that
sources the per-band PSFs and bands directly from the persisted model's
metadata — the same PSFs the deblender used during the fit. The previous
flow re-derived per-band PSFs from the supplied ``MultibandExposure`` at
the blend's stored ``psf_center``; modern ``ScarletBlendData`` no longer
carries ``psf_center`` or ``bands`` attributes (they were dropped during
the scarlet_lite IO refactor), so that code path raised
``AttributeError`` against any model produced after that refactor and
worked only on legacy-zip-read data.

The bare ``model_psf`` parameter is retained but emits a
``FutureWarning``; it will be removed after v31. The original positional
order ``(blendData, model_psf, mCoadd)`` is preserved so existing
positional callers continue to work (with a deprecation warning if they
pass ``model_psf``). To accommodate the new ``model_psf=None`` default,
``mCoadd`` now also defaults to ``None`` at the signature level and is
validated at runtime — calling ``loadBlend`` without ``mCoadd`` raises
``ValueError`` rather than producing a confusing ``AttributeError`` deep
inside the observation construction.

.. warning::

   Migrate to ``loadBlend(blendData, mCoadd=mCoadd, modelData=modelData)``
   to drop the ``FutureWarning`` and prepare for the v31 removal of
   ``model_psf``. In the future mCoadd and modelData will return to
   positional arguments but the new keyword-only signature is a temorary accommodation for the transition.

Per-source blendedness / overlap metrics now populated on the catalog
---------------------------------------------------------------------

The four ``np.float32`` schema fields ``deblend_maxOverlap``,
``deblend_fluxOverlap``, ``deblend_fluxOverlapFraction``, and
``deblend_blendedness`` have always been declared on the deblended-source
schema and have always had their values computed by
``setDeblenderMetrics`` during ``updateCatalogFootprints``, but the
computed values were never copied onto the catalog record. Every
deblended row therefore carried the schema default (``NaN``) for these
four columns, regardless of how blended the source actually was.

The values are now written onto each ``sourceRecord`` alongside the
existing ``deblend_scarletFlux`` and ``deblend_peak_instFlux`` writes, so
catalogs produced through ``updateCatalogFootprints`` carry the
single-band Bosch-2018 blendedness, max-pixel neighbor overlap, total
neighbor-flux overlap, and overlap-as-fraction-of-source-flux for every
deblended source.

.. warning::

   Catalogs produced by an earlier version have ``NaN`` in these four
   columns for every deblended source. Selections that filtered on these
   metrics were silently dropping every row; comparisons that read these
   values for blend quality were undefined. Re-run the deblender (or
   re-run ``updateCatalogFootprints`` over the persisted ``ScarletModelData``
   and existing catalogs) to obtain populated values.

``calculate_update_step`` renamed to ``calculateUpdateStep``
-----------------------------------------------------------

The deconvolution-step helper in
``lsst.meas.extensions.scarlet.deconvolveExposureTask`` was renamed
from ``calculate_update_step`` to ``calculateUpdateStep`` to match
the camelCase convention used elsewhere in the module. The
keyword-argument names follow suit (``min_scale`` → ``minScale``,
``default_scale`` → ``defaultScale``). The snake_case name is
retained as a thin shim that forwards to the new function and emits
a ``FutureWarning`` on every call; it will be removed after v31.

Deprecated ``ScarletDeblendConfig`` fields removed
--------------------------------------------------

Four long-deprecated ``ScarletDeblendConfig`` fields are removed:
``version``, ``morphImage``, ``waveletScales``, and ``sourceModel``.
Their removal had been scheduled in ``deprecated=`` notes since
v29.0; pipelines past that release should already have stopped
setting them.
