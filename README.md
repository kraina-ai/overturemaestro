<p align="center">
  <img width="300" src="https://raw.githubusercontent.com/kraina-ai/overturemaestro/main/docs/assets/logos/overturemaestro_logo.png"><br/>
  <small>Generated using DALL·E 3 model with this prompt: Cute stylized conducting virtuoso using a paper map as music sheet. White background, minimalistic, vector graphics, clean background, encased in a circle. In navy and gold colours. Logo for a python library, should work well as small icon.</small>
</p>

<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/kraina-ai/overturemaestro?logo=apache&logoColor=%23fff">
    <img src="https://img.shields.io/github/checks-status/kraina-ai/overturemaestro/main?logo=GitHubActions&logoColor=%23fff" alt="Checks">
    <a href="https://github.com/kraina-ai/overturemaestro/actions/workflows/ci-dev.yml" target="_blank">
        <img alt="GitHub Workflow Status - DEV" src="https://img.shields.io/github/actions/workflow/status/kraina-ai/overturemaestro/ci-dev.yml?label=build-dev&logo=GitHubActions&logoColor=%23fff">
    </a>
    <a href="https://github.com/kraina-ai/overturemaestro/actions/workflows/ci-prod.yml" target="_blank">
        <img alt="GitHub Workflow Status - PROD" src="https://img.shields.io/github/actions/workflow/status/kraina-ai/overturemaestro/ci-prod.yml?label=build-prod&logo=GitHubActions&logoColor=%23fff">
    </a>
    <a href="https://results.pre-commit.ci/latest/github/kraina-ai/overturemaestro/main" target="_blank">
        <img src="https://results.pre-commit.ci/badge/github/kraina-ai/overturemaestro/main.svg" alt="pre-commit.ci status">
    </a>
    <a href="https://www.codefactor.io/repository/github/kraina-ai/overturemaestro"><img alt="CodeFactor Grade" src="https://img.shields.io/codefactor/grade/github/kraina-ai/overturemaestro?logo=codefactor&logoColor=%23fff"></a>
    <a href="https://app.codecov.io/gh/kraina-ai/overturemaestro/tree/main"><img alt="Codecov" src="https://img.shields.io/codecov/c/github/kraina-ai/overturemaestro?logo=codecov&token=PRS4E02ZX0&logoColor=%23fff"></a>
    <a href="https://pypi.org/project/overturemaestro" target="_blank">
        <img src="https://img.shields.io/pypi/v/overturemaestro?color=%2334D058&label=pypi%20package&logo=pypi&logoColor=%23fff" alt="Package version">
    </a>
    <a href="https://pypi.org/project/overturemaestro" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/overturemaestro.svg?color=%2334D058&logo=python&logoColor=%23fff" alt="Supported Python versions">
    </a>
    <a href="https://pypi.org/project/overturemaestro" target="_blank">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/overturemaestro">
    </a>
</p>

# OvertureMaestro

An open-source tool for reading OvertureMaps data with multiprocessing and additional Quality-of-Life features.

## What is **OvertureMaestro** 🎼🌍?

- Scalable reader for OvertureMaps data.
- Is based on top of `PyArrow`[^1].
- Saves files in the `GeoParquet`[^2] file format for easier integration with modern cloud stacks.
- Filters data based on geometry.
- Can filter data using PyArrow expressions.
- Utilizes multiprocessing for faster data download.
- Utilizes dedicated index of all features in the Overture Maps dataset to download only specific parts based on the geometry filter.
- Utilizes caching to reduce repeatable computations.
- Can be used as Python module as well as a beautiful CLI based on `Typer`[^3].

[^1]: [PyArrow Website](https://arrow.apache.org/docs/python/)
[^2]: [GeoParquet data format](https://geoparquet.org/)
[^3]: [Typer docs](https://typer.tiangolo.com/)

## Installing

### As pure Python module

```
pip install overturemaestro
```

### With beautiful CLI

```
pip install overturemaestro[cli]
```

### Required Python version?

OvertureMaestro supports **Python >= 3.9**

### Dependencies

Required:

- `overturemaps (>=0.8.0)`: Reusing oficial CLI library with dedicated schema related functions

- `pyarrow (>=16.0.0)`: For OvertureMaps GeoParquet dataset wrangling

- `geopandas (>=1.0)`: For returning GeoDataFrames and reading Geo files

- `shapely (>=2.0)`: For parsing WKT and GeoJSON strings and filtering data with STRIndex

- `geoarrow-rust-core (>=0.3.0)`: For transforming Arrow data to Shapely objects

- `duckdb (>=1.1.0)`: For transforming downloaded data to the wide format

- `pooch (>=1.6.0)`: For downloading precalculated dataset indexes

- `rich (>=12.0.0)`: For showing progress bars

- `fsspec (>=2021.04.0)` & `aiohttp (>=3.8.0)`: For accessing AWS S3 datasets in PyArrow and GitHub files for precalculated datasets

- `geopy (>=2.0.0)`: For geocoding of strings

Optional:

- `typer[all] (>=0.9.0)` (click, colorama, rich, shellingham): Required in CLI

- `h3 (>=4.0.0b1)`: For reading H3 strings. Required in CLI

- `s2 (>=0.1.9)`: For transforming S2 indexes into geometries. Required in CLI

- `python-geohash (>=0.8)`: For transforming GeoHash indexes into geometries. Required in CLI

- `scikit-learn (>=1.0)`: For clustering geometries when generating release index. Required for generating release index

- `polars (>=0.20.4)`: For calculating total bounding box from many bounding boxes. Required for generating release index

## Usage

TODO
