# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Modified release index consolidation script

## [0.1.0] - 2024-10-31

### Added

- CLI [#3](https://github.com/kraina-ai/overturemaestro/issues/3)
- Option to filter data with bounding box [#4](https://github.com/kraina-ai/overturemaestro/issues/4)
- Tests for the library [#6](https://github.com/kraina-ai/overturemaestro/issues/6)
- Automatic newest release version loading [#7](https://github.com/kraina-ai/overturemaestro/issues/7)
- Library docs [#2](https://github.com/kraina-ai/overturemaestro/issues/2)
- README content
- Verbosity modes
- Total operation time
- Overloads for the functions typing
- Function for displaying all available release versions
- GitHub Action workflows for docs deployment

### Changed

- Moved location of the pregenerated release indexes to the global cache [#19](https://github.com/kraina-ai/overturemaestro/issues/19)
- Moved `scikit-learn` and `polars` to the dedicated dependency group [#9](https://github.com/kraina-ai/overturemaestro/issues/9)
- Sped up intersection algorithm
- Reduced number of max concurrent connections for parquet files download

### Fixed

- Memory leak during concurrent parquet files download
- Added automatic retry for downloads with 10 retries

## [0.0.3] - 2024-09-08

### Added

- Alternative bounding box related functions for downloading data

## [0.0.2] - 2024-09-06

### Added

- Basic library tests
- CI/CD workflows

### Fixed

- Added missing Pooch and geoarrow-rust-core dependencies
- Cleaned other dependencies
- Changed forward slashes in URLs on Windows

## [0.0.1] - 2024-09-02

### Added

- Release index generation
- Functions for downloading the data using generated indexes
- Function for displaying available theme and type values

[Unreleased]: https://github.com/kraina-ai/overturemaestro/compare/0.1.0...HEAD

[0.1.0]: https://github.com/kraina-ai/overturemaestro/compare/0.0.3...0.1.0

[0.0.3]: https://github.com/kraina-ai/overturemaestro/compare/0.0.2...0.0.3

[0.0.2]: https://github.com/kraina-ai/overturemaestro/compare/0.0.1...0.0.2

[0.0.1]: https://github.com/kraina-ai/overturemaestro/releases/tag/0.0.1
