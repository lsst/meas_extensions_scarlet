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
