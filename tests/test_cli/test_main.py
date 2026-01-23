"""Tests for the main CLI module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from tokenledger.cli import app

runner = CliRunner()


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_command(self) -> None:
        """Test that version command shows version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "TokenLedger" in result.output


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_command_with_defaults(self) -> None:
        """Test serve command starts server with defaults."""
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(app, ["serve"], input="\x03")  # Ctrl+C to exit
            # Server is called but will be interrupted
            assert "Starting TokenLedger server" in result.output

    def test_serve_command_custom_options(self) -> None:
        """Test serve command with custom options."""
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "9000"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 9000

    def test_serve_command_with_reload(self) -> None:
        """Test serve command with reload option."""
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(app, ["serve", "--reload"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["reload"] is True
            # When reload is True, app is passed as string
            assert mock_run.call_args[0][0] == "tokenledger.server:app"


class TestMainApp:
    """Tests for main CLI app."""

    def test_no_args_shows_help(self) -> None:
        """Test that no args shows help (exit code 0 for typer with no_args_is_help)."""
        result = runner.invoke(app, [])
        # With no_args_is_help=True, typer shows help and exits with 0
        assert "TokenLedger" in result.output
        assert "db" in result.output
        assert "serve" in result.output

    def test_help_option(self) -> None:
        """Test --help option."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "TokenLedger" in result.output


class TestMainFunction:
    """Tests for main entry point."""

    def test_main_function_exists(self) -> None:
        """Test that main function exists and is callable."""
        from tokenledger.cli import main

        assert callable(main)
