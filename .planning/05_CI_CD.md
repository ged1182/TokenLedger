# CI/CD Pipeline Design for TokenLedger

## Current State Assessment

### What Exists
- **No CI/CD workflows**: The `.github/workflows/` directory does not exist
- **Development tooling**: Well-configured `pyproject.toml` with:
  - ruff for linting/formatting (>=0.8.0)
  - mypy for type checking (>=1.13.0)
  - pytest with asyncio support and coverage
  - pre-commit hooks configured
- **Multi-Python support**: Project targets Python 3.11, 3.12, 3.13
- **Docker support**: `docker-compose.yml` with Postgres 16 for integration testing
- **Test suite**: Basic test structure exists (`tests/test_pricing.py`, `conftest.py`)
- **Package metadata**: PyPI-ready configuration with hatchling build system

### What's Missing
- GitHub Actions workflows
- Automated testing on PRs
- Release automation
- Security scanning
- Dependency auditing
- Badge generation
- Branch protection rules

---

## Proposed CI Workflow

### Overview
The CI workflow should run on every push and PR to ensure code quality and functionality.

### Workflow: `ci.yml`

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

# Cancel in-progress runs for the same branch
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  PYTHON_VERSION_DEFAULT: "3.11"

jobs:
  # ============================================================================
  # Code Quality Checks (Fast)
  # ============================================================================
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION_DEFAULT }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv pip install --system -e ".[dev]"

      - name: Check formatting
        run: ruff format --check tokenledger tests

      - name: Check linting
        run: ruff check tokenledger tests

  type-check:
    name: Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION_DEFAULT }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv pip install --system -e ".[all,dev]"

      - name: Run mypy
        run: mypy tokenledger

  # ============================================================================
  # Unit Tests (Matrix)
  # ============================================================================
  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv pip install --system -e ".[all,dev]"

      - name: Run tests
        run: pytest -m "not integration" --cov=tokenledger --cov-report=xml --cov-report=term-missing

      - name: Upload coverage to Codecov
        if: matrix.python-version == '3.11'
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage.xml
          fail_ci_if_error: false
          verbose: true

  # ============================================================================
  # Integration Tests (Postgres)
  # ============================================================================
  integration:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: [lint, type-check]  # Only run after basic checks pass

    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: tokenledger
          POSTGRES_PASSWORD: tokenledger
          POSTGRES_DB: tokenledger
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION_DEFAULT }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv pip install --system -e ".[all,dev]"

      - name: Run database migrations
        run: |
          PGPASSWORD=tokenledger psql -h localhost -U tokenledger -d tokenledger -f migrations/001_initial.sql

      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://tokenledger:tokenledger@localhost:5432/tokenledger
        run: pytest -m integration --cov=tokenledger --cov-report=xml

  # ============================================================================
  # Security Scanning
  # ============================================================================
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION_DEFAULT }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install security tools
        run: |
          uv pip install --system bandit[toml] pip-audit

      - name: Run bandit security scan
        run: bandit -r tokenledger -c pyproject.toml || true  # Don't fail initially

      - name: Audit dependencies
        run: |
          uv pip install --system -e ".[all]"
          pip-audit --strict || true  # Don't fail initially

  # ============================================================================
  # Build Verification
  # ============================================================================
  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [test]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION_DEFAULT }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Install build dependencies
        run: uv pip install --system build twine

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
          retention-days: 7

  # ============================================================================
  # Dashboard Build (React)
  # ============================================================================
  dashboard:
    name: Build Dashboard
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: dashboard
    steps:
      - uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: dashboard/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: Run linting
        run: npm run lint || true  # Enable once linting is configured

      - name: Build dashboard
        run: npm run build

      - name: Upload dashboard build
        uses: actions/upload-artifact@v4
        with:
          name: dashboard-dist
          path: dashboard/dist/
          retention-days: 7
```

---

## Proposed Release Workflow

### Overview
Automated release workflow triggered by pushing a version tag (e.g., `v0.1.0`).

### Workflow: `release.yml`

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

permissions:
  contents: write
  id-token: write  # Required for trusted publishing

jobs:
  # ============================================================================
  # Run Full CI First
  # ============================================================================
  ci:
    uses: ./.github/workflows/ci.yml

  # ============================================================================
  # Build and Publish to PyPI
  # ============================================================================
  publish:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: [ci]
    environment:
      name: pypi
      url: https://pypi.org/project/tokenledger/

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build dependencies
        run: pip install build

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # Uses trusted publishing - no API token needed
        # Requires PyPI project to be configured for GitHub OIDC

  # ============================================================================
  # Create GitHub Release
  # ============================================================================
  release:
    name: Create GitHub Release
    runs-on: ubuntu-latest
    needs: [publish]

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for changelog

      - name: Generate changelog
        id: changelog
        uses: orhun/git-cliff-action@v3
        with:
          config: cliff.toml
          args: --latest --strip header
        env:
          OUTPUT: CHANGELOG.md

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          body: ${{ steps.changelog.outputs.content }}
          draft: false
          prerelease: ${{ contains(github.ref, '-alpha') || contains(github.ref, '-beta') || contains(github.ref, '-rc') }}
          files: |
            dist/*.whl
            dist/*.tar.gz
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Workflow: `version-bump.yml` (Manual Version Bumping)

```yaml
# .github/workflows/version-bump.yml
name: Version Bump

on:
  workflow_dispatch:
    inputs:
      bump_type:
        description: 'Type of version bump'
        required: true
        type: choice
        options:
          - patch
          - minor
          - major

permissions:
  contents: write
  pull-requests: write

jobs:
  bump:
    name: Bump Version
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install bump-my-version
        run: pip install bump-my-version

      - name: Configure git
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Bump version
        id: bump
        run: |
          bump-my-version bump ${{ inputs.bump_type }} --commit --tag
          echo "version=$(bump-my-version show current_version)" >> $GITHUB_OUTPUT

      - name: Push changes
        run: |
          git push origin main --tags

      - name: Output new version
        run: echo "Bumped to version ${{ steps.bump.outputs.version }}"
```

---

## Supplementary Configuration Files

### `cliff.toml` (Changelog Generation)

```toml
# cliff.toml - Git Cliff Configuration for Changelog Generation
[changelog]
header = """
# Changelog

All notable changes to this project will be documented in this file.

"""
body = """
{% if version %}\
    ## [{{ version | trim_start_matches(pat="v") }}] - {{ timestamp | date(format="%Y-%m-%d") }}
{% else %}\
    ## [Unreleased]
{% endif %}\
{% for group, commits in commits | group_by(attribute="group") %}
    ### {{ group | striptags | trim | upper_first }}
    {% for commit in commits %}
        - {% if commit.scope %}**{{ commit.scope }}:** {% endif %}\
            {{ commit.message | upper_first }}\
    {% endfor %}
{% endfor %}
"""
footer = """
"""
trim = true

[git]
conventional_commits = true
filter_unconventional = true
split_commits = false

commit_parsers = [
    { message = "^feat", group = "Features" },
    { message = "^fix", group = "Bug Fixes" },
    { message = "^doc", group = "Documentation" },
    { message = "^perf", group = "Performance" },
    { message = "^refactor", group = "Refactoring" },
    { message = "^style", group = "Style" },
    { message = "^test", group = "Testing" },
    { message = "^chore\\(release\\)", skip = true },
    { message = "^chore", group = "Miscellaneous" },
]

filter_commits = false
tag_pattern = "v[0-9].*"
```

### `pyproject.toml` Additions for Version Bumping

```toml
# Add to pyproject.toml

[tool.bumpversion]
current_version = "0.1.0"
commit = true
tag = true
tag_name = "v{new_version}"

[[tool.bumpversion.files]]
filename = "pyproject.toml"
search = 'version = "{current_version}"'
replace = 'version = "{new_version}"'

[[tool.bumpversion.files]]
filename = "tokenledger/__init__.py"
search = '__version__ = "{current_version}"'
replace = '__version__ = "{new_version}"'
```

### `pyproject.toml` Additions for Bandit

```toml
# Add to pyproject.toml

[tool.bandit]
exclude_dirs = ["tests", "examples"]
skips = ["B101"]  # Allow assert statements
```

---

## Branch Protection Recommendations

### Main Branch Protection Rules

Configure these settings in GitHub repository settings under "Branches":

1. **Require a pull request before merging**
   - Require approvals: 1 (or more for teams)
   - Dismiss stale pull request approvals when new commits are pushed
   - Require review from Code Owners (if CODEOWNERS file exists)

2. **Require status checks to pass before merging**
   - Required checks:
     - `lint` - Linting and formatting
     - `type-check` - Type checking with mypy
     - `test (3.11)` - Unit tests on primary Python version
     - `test (3.12)` - Unit tests on Python 3.12
     - `test (3.13)` - Unit tests on Python 3.13
     - `build` - Package build verification
   - Require branches to be up to date before merging

3. **Require conversation resolution before merging**
   - Ensures all code review comments are addressed

4. **Do not allow bypassing the above settings**
   - Even administrators must follow rules

5. **Restrict who can push to matching branches**
   - Only allow merging via pull requests

### Tag Protection Rules

Protect release tags to prevent unauthorized releases:

```yaml
# .github/workflows/tag-protection.yml (informational)
# Configure in repository Settings > Tags > Protected tags
# Pattern: v*
# Allow: Maintainers only
```

---

## Badge Recommendations

### Recommended Badges for README.md

```markdown
# TokenLedger

[![CI](https://github.com/ged1182/tokenledger/actions/workflows/ci.yml/badge.svg)](https://github.com/ged1182/tokenledger/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ged1182/tokenledger/branch/main/graph/badge.svg)](https://codecov.io/gh/ged1182/tokenledger)
[![PyPI version](https://badge.fury.io/py/tokenledger.svg)](https://badge.fury.io/py/tokenledger)
[![Python versions](https://img.shields.io/pypi/pyversions/tokenledger.svg)](https://pypi.org/project/tokenledger/)
[![License: ELv2](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
```

### Badge Descriptions

| Badge | Purpose | Service |
|-------|---------|---------|
| CI | Shows current build status | GitHub Actions |
| codecov | Displays test coverage percentage | Codecov.io |
| PyPI version | Current published version | PyPI |
| Python versions | Supported Python versions | PyPI |
| License | License type | shields.io |
| Code style: ruff | Indicates ruff for linting | shields.io |
| mypy | Indicates type checking | mypy-lang.org |

---

## Dependabot Configuration

### `.github/dependabot.yml`

```yaml
# .github/dependabot.yml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "python"
    commit-message:
      prefix: "chore(deps)"
    groups:
      development:
        patterns:
          - "pytest*"
          - "ruff"
          - "mypy"
          - "pre-commit"
        update-types:
          - "minor"
          - "patch"
      production:
        patterns:
          - "*"
        exclude-patterns:
          - "pytest*"
          - "ruff"
          - "mypy"
          - "pre-commit"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    labels:
      - "dependencies"
      - "github-actions"
    commit-message:
      prefix: "chore(deps)"

  # npm (dashboard)
  - package-ecosystem: "npm"
    directory: "/dashboard"
    schedule:
      interval: "weekly"
      day: "monday"
    labels:
      - "dependencies"
      - "javascript"
    commit-message:
      prefix: "chore(deps)"
```

---

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. **Create `.github/workflows/ci.yml`** - Core CI pipeline
   - Lint + format checks
   - Type checking
   - Unit tests across Python 3.11, 3.12, 3.13
   - Build verification

2. **Create `.github/dependabot.yml`** - Automated dependency updates

3. **Update README.md** - Add CI badge

### Phase 2: Quality Gates (Week 2)
4. **Enable branch protection** - Require CI to pass for merges

5. **Add integration tests** - Tests with real Postgres
   - Update `ci.yml` with Postgres service container

6. **Add Codecov integration** - Coverage reporting
   - Sign up at codecov.io
   - Add `CODECOV_TOKEN` secret

### Phase 3: Security (Week 3)
7. **Add security scanning** - Bandit + pip-audit
   - Add `[tool.bandit]` to pyproject.toml

8. **Configure Dependabot alerts** - Enable in repo settings

### Phase 4: Release Automation (Week 4)
9. **Create `.github/workflows/release.yml`** - Automated releases
   - Configure PyPI trusted publishing

10. **Add `cliff.toml`** - Changelog generation

11. **Add version bumping** - Manual workflow for version management

### Phase 5: Polish (Week 5)
12. **Add all badges** - Full badge collection in README

13. **Create CODEOWNERS** - Define code owners for review

14. **Document release process** - In CONTRIBUTING.md

---

## Files to Create

| File Path | Purpose |
|-----------|---------|
| `.github/workflows/ci.yml` | Main CI pipeline |
| `.github/workflows/release.yml` | Release automation |
| `.github/workflows/version-bump.yml` | Manual version bumping |
| `.github/dependabot.yml` | Dependency updates |
| `cliff.toml` | Changelog generation config |

## Files to Update

| File Path | Changes |
|-----------|---------|
| `pyproject.toml` | Add bandit config, bump-my-version config |
| `README.md` | Add CI/CD badges |
| `tokenledger/__init__.py` | Add `__version__` if not present |

---

## Cost Considerations

All recommended services are free for open source:
- **GitHub Actions**: Free for public repos
- **Codecov**: Free for public repos
- **PyPI**: Free to publish
- **Dependabot**: Free (built into GitHub)

---

## Success Metrics

After implementation, the CI/CD pipeline should provide:

1. **Fast feedback** - Lint/type checks complete in < 2 minutes
2. **Comprehensive coverage** - Tests across all supported Python versions
3. **Reliable releases** - One-command release to PyPI
4. **Security visibility** - Known vulnerabilities flagged automatically
5. **Dependency freshness** - Automated PRs for updates

This CI/CD setup demonstrates professional software engineering practices and is suitable for a portfolio project showcasing DevOps/MLOps skills.
