"""Tests for version consistency and metadata."""

from __future__ import annotations

import re
from importlib.metadata import version

import tokenledger


class TestVersion:
    """Tests for package version."""

    def test_version_exists(self) -> None:
        """Test that __version__ is defined."""
        assert hasattr(tokenledger, "__version__")
        assert tokenledger.__version__ is not None

    def test_version_format(self) -> None:
        """Test that version follows semantic versioning format."""
        # Semantic versioning: MAJOR.MINOR.PATCH with optional pre-release
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$"
        assert re.match(pattern, tokenledger.__version__), (
            f"Version '{tokenledger.__version__}' doesn't match semantic versioning"
        )

    def test_version_matches_metadata(self) -> None:
        """Test that __version__ matches package metadata version."""
        metadata_version = version("tokenledger")
        assert tokenledger.__version__ == metadata_version, (
            f"__version__ ({tokenledger.__version__}) != metadata version ({metadata_version})"
        )


class TestPackageMetadata:
    """Tests for package metadata."""

    def test_package_name(self) -> None:
        """Test that package has correct name in metadata."""
        from importlib.metadata import metadata

        meta = metadata("tokenledger")
        assert meta["Name"] == "tokenledger"

    def test_package_has_author(self) -> None:
        """Test that package has author metadata."""
        from importlib.metadata import metadata

        meta = metadata("tokenledger")
        # Author can be in Author field or Author-email field
        author = meta.get("Author") or meta.get("Author-email")
        assert author is not None

    def test_package_has_license(self) -> None:
        """Test that package has license metadata."""
        from importlib.metadata import metadata

        meta = metadata("tokenledger")
        license_info = meta.get("License") or meta.get("License-Expression")
        assert license_info is not None


class TestPublicAPI:
    """Tests for public API exports."""

    def test_all_exports_exist(self) -> None:
        """Test that all __all__ exports are actually available."""
        for name in tokenledger.__all__:
            assert hasattr(tokenledger, name), f"'{name}' in __all__ but not exported"

    def test_core_exports(self) -> None:
        """Test that core functionality is exported."""
        # Core classes
        assert hasattr(tokenledger, "TokenTracker")
        assert hasattr(tokenledger, "AsyncTokenTracker")
        assert hasattr(tokenledger, "LLMEvent")

        # Configuration
        assert hasattr(tokenledger, "configure")
        assert hasattr(tokenledger, "get_config")

        # Interceptors
        assert hasattr(tokenledger, "patch_openai")
        assert hasattr(tokenledger, "patch_anthropic")
        assert hasattr(tokenledger, "unpatch_openai")
        assert hasattr(tokenledger, "unpatch_anthropic")

        # Decorators
        assert hasattr(tokenledger, "track_llm")
        assert hasattr(tokenledger, "track_cost")

        # Tracker access
        assert hasattr(tokenledger, "get_tracker")
        assert hasattr(tokenledger, "get_async_tracker")
