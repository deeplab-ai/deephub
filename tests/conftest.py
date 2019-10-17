from __future__ import absolute_import

from pathlib import Path

import pytest


@pytest.fixture
def datadir():
    """
    Get the local directory with all files fixtures
    :rtype: Path
    """
    return Path(__file__).resolve().parent / 'datadir'
