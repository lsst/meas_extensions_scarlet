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

import os
import tempfile
import unittest

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

    def test_persistence(self):
        # Test that the model data is persisted correctly
        bundle = pipeline.deblend(
            pipeline.deconvolve(
                pipeline.detect(pipeline.build_image(SCENES["multi-blend"]))
            )
        )
        modelData = bundle.result.scarletModelData
        repo = self._setup_butler()
        bands = modelData.metadata["bands"]
        butler = makeTestCollection(repo, uniqueId="test_run1")
        butler.put(modelData, "scarlet_model_data", dataId={})
        modelData2 = butler.get("scarlet_model_data", dataId={})
        model_psf = modelData.metadata["model_psf"][None, :, :]
        model_psf2 = modelData2.metadata["model_psf"][None, :, :]
        np.testing.assert_almost_equal(model_psf2, model_psf)
        psf = modelData.metadata["psf"]
        psf2 = modelData2.metadata["psf"]
        np.testing.assert_almost_equal(psf2, psf)
        self.assertEqual(len(modelData2.blends), len(modelData.blends))

        for parentId in modelData.blends.keys():
            nChildren = len(modelData.blends[parentId].children)
            self.assertEqual(nChildren, len(modelData2.blends[parentId].children))
            for blendId in modelData.blends[parentId].children:
                blendData1 = modelData.blends[parentId].children[blendId]
                blendData2 = modelData2.blends[parentId].children[blendId]
                self._test_blend(blendData1, blendData2, model_psf, psf, bands)

        for sourceId in modelData.isolated.keys():
            isolatedData1 = modelData.isolated[sourceId]
            isolatedData2 = modelData2.isolated[sourceId]
            self.assertTupleEqual(isolatedData1.origin, isolatedData2.origin)
            np.testing.assert_array_equal(
                isolatedData1.span_array,
                isolatedData2.span_array,
            )

        # Test extracting a single blend
        modelData2 = butler.get("scarlet_model_data", dataId={}, parameters={"blend_id": parentId})
        self.assertEqual(len(modelData2.blends), 1)

        for blendId, blendData1 in modelData.blends[parentId].children.items():
            blendData2 = modelData2.blends[parentId].children[blendId]
            self._test_blend(blendData1, blendData2, model_psf, psf, bands)

        # Test extracting two blends
        modelData2 = butler.get("scarlet_model_data", dataId={}, parameters={"blend_id": [2, 3]})
        self.assertEqual(len(modelData2.blends), 2)
        for parentId in [2, 3]:
            parentData1 = modelData.blends[parentId]
            parentData2 = modelData2.blends[parentId]
            self.assertEqual(len(parentData1.children), len(parentData2.children))
            for blendId in parentData1.children.keys():
                blendData1 = parentData1.children[blendId]
                blendData2 = parentData2.children[blendId]
                self._test_blend(blendData1, blendData2, model_psf, psf, bands)

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


if __name__ == "__main__":
    unittest.main()
