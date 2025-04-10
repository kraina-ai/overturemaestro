# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Option to specify compression and compression_level for result parquet files

### Changed

- Bumped minimal DuckDB version to 1.1.2
- Refactored sorting algorithm and added dedicated compression function
- Refactored components logic and exported it to an external library

### Fixed

- Removed duplicate geo metadata entry in the sorted geoparquet schema
- Compressed number of columns in wide format to integers indexes before sorting

## [0.3.2] - 2025-03-21

### Fixed

- Copy original parquet file schema metadata during geometries sorting

## [0.3.1] - 2025-03-11

### Fixed

- Create a working directory during sorting operation if it doesn't exist
- Pass a working directory to the sorting operation during release index generation

## [0.3.0] - 2025-03-10

### Added

- Option to sort result files by geometry to reduce file size

### Changed

- Default result parquet file compression from `snappy` to `zstd`
- Number of rows in a parquet row group to `100000`

## [0.2.6] - 2025-01-26

### Fixed

- Function detection in `elapsed_time_decorator` for Google Colab environment

## [0.2.5] - 2025-01-22

### Added

- Option to pass list of `hierarchy_depth` values for multiple theme / type pairs
- Info about current theme / type pair to the `HierarchyDepthOutOfBoundsWarning`

## [0.2.4] - 2025-01-19

### Added

- Places hierarchy based on the official taxonomy [#63](https://github.com/kraina-ai/overturemaestro/issues/63)
- Option to change minimal confidence score for places and select only primary category for the wide form transformation [#63](https://github.com/kraina-ai/overturemaestro/issues/63)

### Changed

- Added option to use any non-negative integer as a `hierarchy_depth` value for wide form processing [#64](https://github.com/kraina-ai/overturemaestro/issues/64)
- Shortened hash parts for generated file names to 8 characters per part

### Fixed

- Bug where a constant value has been overwritten instead of being copied before modifying

## [0.2.3] - 2025-01-17

### Fixed

- Changed wide format places definition for older release versions
- Changed get all columns function for wide format places definition
- Bug where code crashed when release index hit zero matches [#11](https://github.com/kraina-ai/overturemaestro/issues/11)

## [0.2.2] - 2025-01-17

### Fixed

- Changed wide format definitions for different release versions

## [0.2.1] - 2025-01-17

### Added

- Wide format release index to precalculate all possible columns [#43](https://github.com/kraina-ai/overturemaestro/issues/43)
- Flag `include_all_possible_columns` to keep or prune empty columns [#43](https://github.com/kraina-ai/overturemaestro/issues/43)
- `overturemaestro.advanced_functions.wide_form.get_all_possible_column_names` for getting a list of all possible column names [#46](https://github.com/kraina-ai/overturemaestro/issues/46)
- `overturemaestro.cache.clear_cache` function for clearing local release index cache from the API

## [0.2.0] - 2025-01-16

### Added

- Automatic total time wrapper decorator to aggregate nested function calls
- Parameter `columns_to_download` for selecting columns to download from the dataset [#23](https://github.com/kraina-ai/overturemaestro/issues/23)
- Option to pass a list of pyarrow filters and columns for download for each theme type pair when downloading multiple datasets at once
- Automatic columns detection in pyarrow filters when passing `columns_to_download`
- New `advanced_functions` module with a `wide` format for machine learning purposes [#38](https://github.com/kraina-ai/overturemaestro/issues/38)

### Fixed

- Replaced urllib HTTPError with requests HTTPError in release index download functions

### Changed

- Refactored available release versions caching [#24](https://github.com/kraina-ai/overturemaestro/issues/24)
- Removed hive partitioned parquet schema columns from GeoDataFrame loading

### Deprecated

- Nested fields in PyArrow filter in CLI is now expected to be separated by a dot, not a comma [#22](https://github.com/kraina-ai/overturemaestro/issues/22)

## [0.1.2] - 2024-12-17

### Added

- Option to pass max number of workers for downloading the data [#30](https://github.com/kraina-ai/overturemaestro/issues/30)

## [0.1.1] - 2024-11-24

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

[Unreleased]: https://github.com/kraina-ai/overturemaestro/compare/0.3.2...HEAD

[0.3.2]: https://github.com/kraina-ai/overturemaestro/compare/0.3.1...0.3.2

[0.3.1]: https://github.com/kraina-ai/overturemaestro/compare/0.3.0...0.3.1

[0.3.0]: https://github.com/kraina-ai/overturemaestro/compare/0.2.6...0.3.0

[0.2.6]: https://github.com/kraina-ai/overturemaestro/compare/0.2.5...0.2.6

[0.2.5]: https://github.com/kraina-ai/overturemaestro/compare/0.2.4...0.2.5

[0.2.4]: https://github.com/kraina-ai/overturemaestro/compare/0.2.3...0.2.4

[0.2.3]: https://github.com/kraina-ai/overturemaestro/compare/0.2.2...0.2.3

[0.2.2]: https://github.com/kraina-ai/overturemaestro/compare/0.2.1...0.2.2

[0.2.1]: https://github.com/kraina-ai/overturemaestro/compare/0.2.0...0.2.1

[0.2.0]: https://github.com/kraina-ai/overturemaestro/compare/0.1.2...0.2.0

[0.1.2]: https://github.com/kraina-ai/overturemaestro/compare/0.1.1...0.1.2

[0.1.1]: https://github.com/kraina-ai/overturemaestro/compare/0.1.0...0.1.1

[0.1.0]: https://github.com/kraina-ai/overturemaestro/compare/0.0.3...0.1.0

[0.0.3]: https://github.com/kraina-ai/overturemaestro/compare/0.0.2...0.0.3

[0.0.2]: https://github.com/kraina-ai/overturemaestro/compare/0.0.1...0.0.2

[0.0.1]: https://github.com/kraina-ai/overturemaestro/releases/tag/0.0.1
