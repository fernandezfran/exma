#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021-2023, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import exma.lib

# ============================================================================
# TESTS
# ============================================================================


def test_docs():
    """Test the docs."""
    assert isinstance(exma.lib.__doc__, str)
