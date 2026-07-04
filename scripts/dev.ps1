<#
.SYNOPSIS
    Developer convenience wrapper for common project tasks (PowerShell).
.DESCRIPTION
    Mirrors the Unix Makefile targets for Windows users who prefer PowerShell.
    Every subcommand is a thin wrapper over the same tools invoked by CI, so
    "green locally" implies "green on GitHub Actions".
.PARAMETER Task
    One of: install, install-dev, lint, format, typecheck, test, coverage, ci, clean.
.EXAMPLE
    .\scripts\dev.ps1 ci
.EXAMPLE
    .\scripts\dev.ps1 test
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "help", "install", "install-dev", "lint", "format",
        "typecheck", "test", "coverage", "ci", "clean"
    )]
    [string] $Task = "help"
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

function Invoke-Step {
    param([string] $Description, [scriptblock] $Action)
    Write-Host ""
    Write-Host "=> $Description" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Description (exit code $LASTEXITCODE)"
    }
}

switch ($Task) {
    "help" {
        Write-Host "Available tasks:"
        Write-Host "  install       Install runtime dependencies only"
        Write-Host "  install-dev   Install runtime + development dependencies"
        Write-Host "  lint          Run ruff linter and format check"
        Write-Host "  format        Auto-format code with ruff"
        Write-Host "  typecheck     Run mypy in strict mode"
        Write-Host "  test          Run pytest verbosely"
        Write-Host "  coverage      Run tests with branch coverage and HTML report"
        Write-Host "  ci            Run the full CI pipeline locally"
        Write-Host "  clean         Remove build, cache and coverage artifacts"
    }
    "install" {
        Invoke-Step "Installing runtime dependencies" { pip install -r requirements.txt }
    }
    "install-dev" {
        Invoke-Step "Installing dev dependencies" { pip install -r requirements-dev.txt }
    }
    "lint" {
        Invoke-Step "Ruff lint"          { ruff check . }
        Invoke-Step "Ruff format check"  { ruff format --check . }
    }
    "format" {
        Invoke-Step "Ruff format"        { ruff format . }
        Invoke-Step "Ruff auto-fix lint" { ruff check --fix . }
    }
    "typecheck" {
        Invoke-Step "Mypy (strict)" { mypy . }
    }
    "test" {
        Invoke-Step "Pytest" { pytest -v }
    }
    "coverage" {
        Invoke-Step "Pytest with coverage" {
            pytest --cov=core --cov=utils --cov=agents --cov-branch `
                   --cov-report=term-missing --cov-report=html
        }
    }
    "ci" {
        Invoke-Step "Ruff lint"          { ruff check . }
        Invoke-Step "Ruff format check"  { ruff format --check . }
        Invoke-Step "Mypy (strict)"      { mypy . }
        Invoke-Step "Pytest"             { pytest -v }
    }
    "clean" {
        $paths = @(
            ".pytest_cache", ".mypy_cache", ".ruff_cache",
            "htmlcov", ".coverage", "coverage.xml"
        )
        foreach ($p in $paths) {
            if (Test-Path $p) {
                Remove-Item -Recurse -Force -LiteralPath $p
                Write-Host "Removed $p"
            }
        }
        Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" |
            ForEach-Object {
                Remove-Item -Recurse -Force -LiteralPath $_.FullName
                Write-Host "Removed $($_.FullName)"
            }
    }
}
