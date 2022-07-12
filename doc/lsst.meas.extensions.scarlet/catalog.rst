.. _catalog_deblending:

====================
Deblending a Catalog
====================

Because *scarlet* (and *scarlet lite*) use different data objects for images, convolutions, and source records, it is recommended to use convenience methods in the :py:mod:`~lsst.meas.extensions.scarlet` package to run scarlet on images and catalogs derived from the science pipelines.
This document will describe the input and output data products, configuration settings, and recommended methods for displaying and analyzing both scarlet and catalog results using the multi-band deblender.

.. _inputs:

Inputs
------

In the DM science pipelines a ``calexp`` exposore is made in each band, and re-projected onto the same pixel grid, however any set of exposures that are re-projected onto the same pixel grid can be used as an input to :py:class:`~lsst.meas.extensions.scarlet.ScarletDeblendTask`, the current multi-band deblender.
For example, a set of images in the same band, taken at different times, can be used, provided they all use the same WCS.
In addition to the input exposures, the only other input required by the multi-band deblender is a source catalog.
This catalog should only contain parent source records, where each parent contains a footprint with one or more detected peaks and each peak is modeled as a source in the blend.
In the DM Science Pipelines this catalog is created in two steps.
First :py:class:`~lsst.pipe.tasks.multiBand.DetectCoaddSourcesTask` is run on the coadd in each band, producing a set of source catalogs.
Next :py:class:`~lsst.pipe.tasks.mergeDetections.MergeDetectionsTask` is run to merge the source catalogs into a single catalog that is used as an input to the deblender.

.. _config:

Configuration
-------------

The :py:class:`~lsst.meas.extensions.scarlet.ScarletDeblendConfig` object contains the configuration for :py:class:`~lsst.meas.extensions.scarlet.ScarletDeblendTask`.
For most users it is recommended to use the default settings for all configuration options, with a few exceptions.

The default settings assume that downstream the user will be using the scarlet models to re-distribute the flux from the input image in each band (see :ref:`re-apportion`).
If instead the user intends to use the scarlet models themselves to make measurements, it is recommended to decrease :attr:`~lsst.meas.extensions.scarlet.ScarletDeblendConfig.relativeError` to around ``1e-4``, and possibly also increase :attr:`~lsst.meas.extensions.scarlet.ScarletDeblendConfig.maxIter` to a higher value.
This will increase the overall runtime but ensures that the model has fully converged (see the results in :ref:`scarlet_restart`).

If you are running into memory issues there are a couple of different ways to address this.
Turning off :attr:`~lsst.meas.extensions.scarlet.ScarletDeblendConfig.processSingles` will cause the deblender to skip isolated sources.
If you are re-distributing flux this is reasonable because re-distributing flux for an isolated source will just return the same thing as extracting the image data from the footprint, so there is no need to run the deblender on those parent records at all (they are currently run for diagnostic purposes).

Another place to save memory is by adjusting when to use spectrum initialization (see :ref:`model_init`).
The easiest way to do this is to set :attr:`~lsst.meas.extensions.scarlet.ScarletDeblendConfig.setSpectra` to ``False``, however in general that is not recommended.
The model will converge faster, often to a better log likelihood, when the spectra are initialized properly.
But because it requires building an array that is ``number of source components × number of bands × width × height)``, for large blends this can cause the system to run out of memory.
Internally :py:class:`~lsst.meas.extensions.scarlet.ScarletDeblendTask` checks to see if ``number of peaks × width × height < ScarletDeblendConfig.maxSpectrumCutoff``.
If it is, then the spectra are initialized as described in :ref:`model_init`, otherwise a less accurate estimate is made for the SED of each component independently.
So lowering :attr:`~lsst.meas.extensions.scarlet.ScarletDeblendConfig.maxSpectrumCutoff` can be used to skip spectrum initialization for increasingly smaller blends/number of peaks.

The last place to save memory is to skip very large blends altogether.
This is not ideal, as it means that a large section of an image will not be deblended at all, however in some cases this is necessary due to memory constraints.
This can be done by setting the upper limits for :attr:`~lsst.meas.extensions.scarlet.ScarletDeblendConfig.maxNumberOfPeaks`, :attr:`~lsst.meas.extensions.scarlet.ScarletDeblendConfig.maxFootprintArea`, :attr:`~lsst.meas.extensions.scarlet.ScarletDeblendConfig.maxAreaTimesPeaks`, and :attr:`~lsst.meas.extensions.scarlet.ScarletDeblendConfig.maxFootprintSize`.

.. _output:

Data products
-------------

The output of :py:class:`~lsst.meas.extensions.scarlet.ScarletDeblendTask` is a tuple that contains a single source catalog and a :py:class:`~lsst.meas.extensions.scarlet.io.ScarletModelData` object that contains a persistable model for all of the blends in the catalog.
When running :py:class:`~lsst.pipe.tasks.multiband.DeblendCoaddSourcesTask` the source catalog is saved as ``deepCoadd_deblendedCatalog`` and the model data as ``deepCoadd_scarletModelData``.

.. _reconstruction:

Reconstructing Footprints
-------------------------

The ``deepCoadd_deblenderCatalog`` that is persisted as described in :ref:`output` contains all of the band-agnostic information about all of the parents and deblended sources.
This section will describe how to load a single band catalog as used in the science pipelines for measurement, including footprints for all deblended sources, while :ref:`scarlet_models` will describe how to access the scarlet models themselves for analysis.
To calculate any statistics or measurements that don't require flux information from a given band, the deblended catalog can be quickly retrieved using

.. code-block:: python

    from lsst.daf.butler import Butler
    # Initialize the butler
    butler = Butler("/repo/main", skymap="hsc_rings_v1", collections=collections)
    # Load the deblender output catalog
    catalog = butler.get("deepCoadd_deblendedCatalog", tract=tract, patch=patch)

where ``collections`` is a list of the collections used to retrieve the data, ``skymap`` is the name of the sky map to use, ``tract`` is the name of the desired tract, and ``patch`` is the desired patch.

In order to retrieve flux measurements and footprints we must also load the scarlet models and attach them to the catalogs.
If we are not using the scarlet models to re-distribute the flux from an exposure then we can simply attach the footprints and flux measurements in a given ``band`` using

.. code-block:: python

    # Load the scarlet models for the catalog
    modelData = butler.get("deepCoadd_scarletModelData", tract=tract, patch=patch)
    # Load the PSF model
    psfModel = butler.get("deepCoadd_calexp.psf", tract=tract, patch=patch, band=band)
    # Update the footprints for all of the deblended sources.
    modelData.updateCatalogFootprints(catalog, band=band, psfModel=psfModel, removeScarletData=True)

Notice that we had to load the PSF from the exposure in order to generate the footprints and measurements.
This is because the scarlet models exist in a partially deconvolved space (see :ref:`basic_scarlet_model`) and needs to be convolved with the difference kernel between the Gaussian PSF used in the scarlet model and the PSF of the exposure.
Setting the ``removeScarletData`` parameter to ``True`` ensures that each record is removed from the ``modelData`` once it has been converted into a :py:class:`~lsst.afw.detection.HeavyFootprint` in order to save memory.
If you want to keep the scarlet models in memory as well, set this option to ``False``.
The ``catalog`` can now be used for any downstream measurement tasks as if the footprint data had always been there.

Unsurprisingly, if you want to use the scarlet models to redistribute the flux from an observed image, then the exposure must also be loaded and included in the update

.. code-block:: python

    # Re-load the scarlet models for the catalog, since it was removed above
    modelData = butler.get("deepCoadd_scarletModelData", tract=tract, patch=patch)
    # Load the PSF model
    psfModel = butler.get("deepCoadd_calexp.psf", tract=tract, patch=patch, band=band)
    # Load the observed image
    image = butler.get("deepCoadd_calexp.image", tract=tract, patch=patch, band=band)
    # Update the footprints for all of the deblended sources.
    modelData.updateCatalogFootprints(catalog, band=band, psfModel=psfModel,redistributeImage=image)

where ``redistributeImage`` is the image used to re-distribute the flux according to the scarlet models.
As before, the ``catalog`` can now be used for measurements in the science pipelines.
