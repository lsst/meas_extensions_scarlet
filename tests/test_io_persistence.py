# This file is part of meas_extensions_scarlet.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Butler persistence tests for ``LsstScarletModelData``.

Round-trips the deblender's on-disk model storage class plus two
back-compatibility shims (a v1.0.0 ``LsstScarletModelData`` ingest and
a v0 ``ScarletModelData`` ingest with a storage-class override). The
deblend that supplies ``modelData`` comes from the cached pipeline
stages in ``pipeline.py`` so that this file does not depend on
``test_deblend.py``'s ad-hoc setup.
"""

import io
import json
import os
import tempfile
import unittest
import zipfile

import lsst.daf.butler
import lsst.meas.extensions.scarlet as mes
import lsst.scarlet.lite
import lsst.utils.tests
import numpy as np
from lsst.daf.butler import (
    Butler,
    Config,
    DatasetRef,
    DatasetType,
    FileDataset,
    StorageClass,
)
from lsst.daf.butler.tests import makeTestCollection, makeTestRepo

import pipeline
from scenes import SCENES

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class TestIoPersistence(lsst.utils.tests.TestCase):
    """Butler put/get and legacy-model tests for
    ``LsstScarletModelData`` storage in
    ``lsst.meas.extensions.scarlet.io``.
    """

    def _persist_modelData(self):
        # Set up a butler with the multi-blend modelData written into
        # it. Sets ``self.modelData``, ``self.model_psf``, ``self.psf``,
        # ``self.bands``, and ``self.butler`` for use by the three
        # put/get tests below. ``pipeline.deblend`` is memoized per
        # scene + config, so the deblend itself is computed once per
        # process even though this helper runs per test.
        bundle = pipeline.deblend(
            pipeline.deconvolve(
                pipeline.detect(pipeline.build_image(SCENES["multi-blend"]))
            )
        )
        self.modelData = bundle.result.scarletModelData
        self.bands = self.modelData.metadata["bands"]
        self.model_psf = self.modelData.metadata["model_psf"][None, :, :]
        self.psf = self.modelData.metadata["psf"]
        repo = self._setup_butler()
        self.butler = makeTestCollection(repo, uniqueId="test_run1")
        self.butler.put(self.modelData, "scarlet_model_data", dataId={})

    def test_butler_put_get_roundtrip(self):
        """A butler ``put`` then ``get`` (no parameters) preserves
        the full ``LsstScarletModelData``.

        Checks ``model_psf`` and ``psf`` metadata, the blend count
        and per-blend children (compared via ``_test_blend``), and the
        isolated-source origins and span arrays.
        """
        self._persist_modelData()
        modelData2 = self.butler.get("scarlet_model_data", dataId={})

        np.testing.assert_almost_equal(
            modelData2.metadata["model_psf"][None, :, :], self.model_psf
        )
        np.testing.assert_almost_equal(modelData2.metadata["psf"], self.psf)
        self.assertEqual(len(modelData2.blends), len(self.modelData.blends))

        for parentId in self.modelData.blends.keys():
            nChildren = len(self.modelData.blends[parentId].children)
            self.assertEqual(nChildren, len(modelData2.blends[parentId].children))
            for blendId in self.modelData.blends[parentId].children:
                blendData1 = self.modelData.blends[parentId].children[blendId]
                blendData2 = modelData2.blends[parentId].children[blendId]
                self._test_blend(blendData1, blendData2, self.model_psf, self.psf, self.bands)

        for sourceId in self.modelData.isolated.keys():
            isolatedData1 = self.modelData.isolated[sourceId]
            isolatedData2 = modelData2.isolated[sourceId]
            self.assertTupleEqual(isolatedData1.origin, isolatedData2.origin)
            np.testing.assert_array_equal(
                isolatedData1.span_array,
                isolatedData2.span_array,
            )

    def test_butler_get_single_blend_parameter(self):
        """``parameters={'blend_id': id}`` returns exactly that one blend.

        The returned modelData contains only the requested parent and
        its children are bit-identical (via ``_test_blend``) to the
        original.
        """
        self._persist_modelData()
        parentId = next(iter(self.modelData.blends))

        modelData2 = self.butler.get(
            "scarlet_model_data", dataId={}, parameters={"blend_id": parentId}
        )

        self.assertEqual(len(modelData2.blends), 1)
        self.assertIn(parentId, modelData2.blends)
        for blendId, blendData1 in self.modelData.blends[parentId].children.items():
            blendData2 = modelData2.blends[parentId].children[blendId]
            self._test_blend(blendData1, blendData2, self.model_psf, self.psf, self.bands)

    def test_butler_get_multiple_blend_parameter(self):
        """``parameters={'blend_id': [...]}`` returns exactly the listed
        blends.

        Picks the first two parent IDs from the multi-blend scene so the
        test does not hardcode specific catalog IDs (which depend on
        detection ordering).
        """
        self._persist_modelData()
        blendIds = list(self.modelData.blends.keys())[:2]

        modelData2 = self.butler.get(
            "scarlet_model_data", dataId={}, parameters={"blend_id": blendIds}
        )

        self.assertEqual(len(modelData2.blends), len(blendIds))
        for parentId in blendIds:
            parentData1 = self.modelData.blends[parentId]
            parentData2 = modelData2.blends[parentId]
            self.assertEqual(len(parentData1.children), len(parentData2.children))
            for blendId in parentData1.children.keys():
                blendData1 = parentData1.children[blendId]
                blendData2 = parentData2.children[blendId]
                self._test_blend(blendData1, blendData2, self.model_psf, self.psf, self.bands)

    def test_legacy_model(self):
        repo = self._setup_butler()
        storageClass = StorageClass(
            "LsstScarletModelData",
            pytype=mes.io.LsstScarletModelData,
        )
        datasetType = DatasetType(
            "old_scarlet_model_data",
            dimensions=(),
            storageClass=storageClass,
            universe=repo.dimensions,
        )
        ref = DatasetRef(
            datasetType,
            run="test_ingestion",
            dataId={},
        )
        dataset = FileDataset(
            path=os.path.join(TESTDIR, "data", "v29_models.json"),
            formatter="lsst.daf.butler.formatters.json.JsonFormatter",
            refs=[ref],
        )

        # Ingest the legacy model into the butler
        butler = makeTestCollection(repo, uniqueId="ingestion")
        repo.registry.registerDatasetType(datasetType)
        butler.ingest(dataset)

        model = butler.get("old_scarlet_model_data", dataId={})
        self.assertEqual(len(model.blends), 2)
        # The pre-``metadata`` archive stored the model PSF as the
        # top-level ``psf`` / ``psfShape`` entries. The legacy
        # migration must promote those into ``metadata['model_psf']``
        # (numpy array, reconstructed via ``array_keys``) so
        # downstream consumers see the same shape as a modern model.
        # Regression test for finding IO-17 of
        # ``audits/audit-2026-05-05.md``.
        self.assertIsNotNone(model.metadata)
        self.assertIn("model_psf", model.metadata)
        self.assertIsInstance(model.metadata["model_psf"], np.ndarray)
        self.assertEqual(model.metadata["model_psf"].shape, (15, 15))
        self.assertNotIn("psfShape", model.metadata)

        test = butler.get("old_scarlet_model_data", dataId={}, parameters={"blend_id": 3495976385350991873})
        self.assertEqual(len(test.blends), 1)

    def test_older_legacy_model(self):
        repo = self._setup_butler()
        oldStorageClass = StorageClass(
            "ScarletModelData",
            pytype=lsst.scarlet.lite.io.ScarletModelData,
        )
        oldDatasetType = DatasetType(
            "old_scarlet_model_data",
            dimensions=(),
            storageClass=oldStorageClass,
            universe=repo.dimensions,
        )
        ref = DatasetRef(
            oldDatasetType,
            run="test_ingestion",
            dataId={},
        )
        dataset = FileDataset(
            path=os.path.join(TESTDIR, "data", "v29_models.json"),
            formatter="lsst.daf.butler.formatters.json.JsonFormatter",
            refs=[ref],
        )

        # Ingest the legacy model into the butler
        butler = makeTestCollection(repo, uniqueId="ingestion")
        repo.registry.registerDatasetType(oldDatasetType)
        butler.ingest(dataset)

        # Load the base repo config from the repository
        base_config = Config(os.path.join(self.repo_dir, "butler.yaml"))

        # Load the storage class override config
        override_path = os.path.join(
            os.path.dirname(lsst.daf.butler.__file__),
            "configs",
            "storageClasses.yaml"
        )
        override_config = Config(override_path)

        # Merge the configs (update base with override)
        base_config.update(override_config)

        # Create Butler with the merged config
        # The config now contains both the repo info and
        # the storage class overrides
        newButler = Butler.from_config(base_config, collections=butler.collections)

        model = newButler.get("old_scarlet_model_data", dataId={}, storageClass="LsstScarletModelData")
        self.assertEqual(len(model.blends), 2)
        self.assertEqual(len(model.isolated), 0)

    def test_read_legacy_zip_without_metadata(self):
        """``read_scarlet_model`` reads a legacy-format zip that has no
        ``metadata`` entry.

        Legacy archives store the model PSF as top-level ``psf`` /
        ``psfShape`` entries instead of a ``metadata`` entry.
        ``zipfile.ZipFile.open`` raises ``KeyError`` (not ``ValueError``)
        for a missing entry, so the legacy fallback was unreachable and
        such archives crashed on read. Regression test for finding C-3
        of the ``audits/audit-2026-05-05.md`` audit; also pins the IO-17
        fix that the legacy load now produces a ``metadata['model_psf']``
        numpy array.
        """
        bundle = pipeline.deblend(
            pipeline.deconvolve(
                pipeline.detect(pipeline.build_image(SCENES["multi-blend"]))
            )
        )
        jm = bundle.result.scarletModelData.as_dict()

        # Repackage the model in the legacy layout: one entry per blend
        # plus a top-level model PSF, and crucially no ``metadata`` entry.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for blendId, blendData in jm["blends"].items():
                zf.writestr(str(blendId), json.dumps(blendData))
            model_psf = jm["metadata"]["model_psf"]
            model_psf_shape = list(np.asarray(model_psf).shape)
            zf.writestr("psf", json.dumps(model_psf))
            zf.writestr("psfShape", json.dumps(model_psf_shape))
        buf.seek(0)

        model = mes.io.utils.read_scarlet_model(buf)
        self.assertEqual(len(model.blends), len(jm["blends"]))
        self.assertIsNotNone(model.metadata)
        self.assertIn("model_psf", model.metadata)
        self.assertIsInstance(model.metadata["model_psf"], np.ndarray)
        self.assertEqual(
            list(model.metadata["model_psf"].shape), model_psf_shape
        )

    def _test_blend(self, blendData1, blendData2, model_psf, psf, bands):
        # Test that two ScarletBlendData objects are equal
        # up to machine precision.
        self.assertTupleEqual(blendData1.origin, blendData2.origin)
        self.assertEqual(len(blendData1.sources), len(blendData2.sources))

        # Test that the two blends are equal up to machine precision
        # once converted into scarlet lite Blend objects.
        blend1 = blendData1.minimal_data_to_blend(
            model_psf,
            psf,
            bands,
            dtype=np.float32,
        )
        blend2 = blendData2.minimal_data_to_blend(
            model_psf,
            psf,
            bands,
            dtype=np.float32,
        )
        np.testing.assert_almost_equal(blend1.get_model().data, blend2.get_model().data)

    def _setup_butler(self):
        # Initialize a Butler to test persistence
        repo_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.repo_dir = repo_dir.name
        self.addCleanup(tempfile.TemporaryDirectory.cleanup, repo_dir)
        config = Config()
        config["datastore", "cls"] = "lsst.daf.butler.datastores.fileDatastore.FileDatastore"
        repo = makeTestRepo(repo_dir.name, config=config)
        storageClass = StorageClass(
            "LsstScarletModelData",
            pytype=mes.io.LsstScarletModelData,
            parameters=('blend_id',),
            delegate="lsst.meas.extensions.scarlet.io.ScarletModelDelegate",
        )
        datasetType = DatasetType(
            "scarlet_model_data",
            dimensions=(),
            storageClass=storageClass,
            universe=repo.dimensions,
        )
        repo.registry.registerDatasetType(datasetType)
        return repo


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
