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

"""Butler persistence tests for ``LsstScarletModelData``."""


import os
import tempfile
import unittest

import lsst.meas.extensions.scarlet as mes
import lsst.utils.tests
from lsst.daf.butler import (
    Config,
    DatasetRef,
    DatasetType,
    FileDataset,
    StorageClass,
)
from lsst.daf.butler.tests import makeTestCollection, makeTestRepo

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class TestIoPersistence(lsst.utils.tests.TestCase):
    """Butler put/get and legacy-model tests for
    ``LsstScarletModelData`` storage in
    ``lsst.meas.extensions.scarlet.io``.
    """

    def test_lsst_scarlet_model_conversion(self):
        """Test converting an LsstScarletModelData to a ScarletModelData
        (lossily) via the Butler.
        """
        repo = self._setup_butler()
        newDatasetType = DatasetType(
            "new_scarlet_model_data",
            dimensions=(),
            storageClass="LsstScarletModelData",
            universe=repo.dimensions,
        )
        ref = DatasetRef(
            newDatasetType,
            run="test_ingestion",
            dataId={},
        )
        dataset = FileDataset(
            path=os.path.join(TESTDIR, "data", "v29_models.json"),
            formatter="lsst.daf.butler.formatters.json.JsonFormatter",
            refs=[ref],
        )
        butler = makeTestCollection(repo, uniqueId="ingestion")
        repo.registry.registerDatasetType(newDatasetType)
        butler.ingest(dataset)
        model = butler.get("new_scarlet_model_data", dataId={}, storageClass="ScarletModelData")
        self.assertIsInstance(model, lsst.scarlet.lite.io.ScarletModelData)
        self.assertEqual(len(model.blends), 2)

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
