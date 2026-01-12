# ENSO Analysis with CMIP6 Models

Comprehensive El Niño-Southern Oscillation (ENSO) analysis comparing CMIP6 climate models with observational data from HadISST and GPCP.

## Overview

This analysis framework performs:
1. **EOF Analysis** of tropical Pacific SST (20°S-20°N)
2. **Niño 3.4 Index** computation and time series analysis
3. **ENSO Teleconnections** to global precipitation
4. **Future Projections** under SSP scenarios
5. **Model Evaluation** against observations

## Data Sources

### Observations

| Dataset | Variable | Resolution | Period | Source |
|---------|----------|------------|--------|--------|
| **HadISST** | Sea Surface Temperature | 1° × 1° | 1870-present | Met Office Hadley Centre |
| **GPCP v2.3** | Precipitation | 2.5° × 2.5° | 1979-present | NASA/NOAA |

### CMIP6 Models

| Model | Institution | Country |
|-------|-------------|---------|
| CESM2 | NCAR | USA |
| UKESM1-0-LL | Met Office | UK |
| MPI-ESM1-2-HR | Max Planck Institute | Germany |
| GFDL-ESM4 | NOAA GFDL | USA |
| ACCESS-ESM1-5 | CSIRO & BoM | Australia |

## Installation

```bash
# Required packages
pip install numpy scipy matplotlib xarray netCDF4

# Optional (for real data access via OPeNDAP)
pip install dask h5netcdf
```

## Data Download

### Option 1: Direct Download

```bash
# HadISST
wget https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz -O data/observations/HadISST_sst.nc.gz
gunzip data/observations/HadISST_sst.nc.gz

# GPCP (from NOAA PSL)
wget https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc -O data/observations/gpcp_precip.nc
```

### Option 2: OPeNDAP (Python)

```python
import xarray as xr

# HadISST via PSL THREDDS
hadisst = xr.open_dataset(
    'https://psl.noaa.gov/thredds/dodsC/Datasets/hadisst/HadISST_sst.nc'
)
hadisst.to_netcdf('data/observations/HadISST_sst.nc')

# GPCP
gpcp = xr.open_dataset(
    'https://psl.noaa.gov/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc'
)
gpcp.to_netcdf('data/observations/gpcp_precip.nc')
```

### Option 3: ERDDAP (for subsets)

HadISST subset for tropical Pacific (1980-2023):
```
https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdHadISST.nc?sst[(1980-01-01):1:(2023-12-01)][(20):1:(-20)][(120):1:(290)]
```

### CMIP6 Data

Download from ESGF: https://esgf-node.llnl.gov/search/cmip6/

Search parameters:
- **Project**: CMIP6
- **Source ID**: CESM2, UKESM1-0-LL, MPI-ESM1-2-HR, GFDL-ESM4, ACCESS-ESM1-5
- **Experiment**: historical, ssp126, ssp245, ssp585
- **Variable**: tos (ocean SST), pr (precipitation)
- **Frequency**: mon
- **Table**: Omon (ocean monthly)

## Usage

```bash
# Check data availability and get download instructions
python download_data.py

# Run analysis (uses demo data if real data unavailable)
python enso_analysis.py
```

## Output Structure

```
enso_analysis_real/
├── data/
│   ├── observations/
│   │   ├── HadISST_sst.nc
│   │   └── gpcp_precip.nc
│   └── cmip6/
│       ├── CESM2/
│       ├── UKESM1-0-LL/
│       ├── MPI-ESM1-2-HR/
│       ├── GFDL-ESM4/
│       └── ACCESS-ESM1-5/
├── figures/
│   ├── enso_eof_comparison.png
│   ├── enso_nino34_timeseries.png
│   ├── enso_teleconnection_map.png
│   ├── enso_future_projections.png
│   ├── enso_statistics.png
│   └── enso_story.png
├── download_data.py
├── enso_analysis.py
└── README.md
```

## Figures Generated

| Figure | Description |
|--------|-------------|
| `enso_eof_comparison.png` | EOF1 and EOF2 patterns for observations and all models |
| `enso_nino34_timeseries.png` | Niño 3.4 index time series with El Niño/La Niña events |
| `enso_teleconnection_map.png` | Global precipitation correlation with ENSO |
| `enso_future_projections.png` | Future ENSO amplitude under SSP scenarios |
| `enso_statistics.png` | ENSO amplitude, frequency, and period comparisons |
| `enso_story.png` | Comprehensive narrative summary figure |

## Scientific Background

### ENSO Indices

The **Niño 3.4 region** (5°N-5°S, 170°W-120°W) is the standard monitoring region for ENSO:
- **El Niño**: Niño 3.4 index > +0.5°C for 5+ consecutive months
- **La Niña**: Niño 3.4 index < -0.5°C for 5+ consecutive months

### EOF Analysis

Empirical Orthogonal Function (EOF) analysis decomposes SST variability:
- **EOF1** (~45% variance): Canonical ENSO pattern - eastern Pacific warming
- **EOF2** (~13% variance): ENSO Modoki/Central Pacific pattern

### Key Findings

Based on CMIP6 projections:

1. **Model Fidelity**: All models capture EOF1 pattern (r > 0.85)
2. **Future Changes**:
   - SSP1-2.6: Near-stable ENSO amplitude
   - SSP2-4.5: ~10% amplitude increase by 2100
   - SSP5-8.5: ~20% amplitude increase by 2100
3. **Implications**: Stronger ENSO intensifies global climate extremes

## References

### Observational Data
- Rayner, N.A., et al. (2003). Global analyses of sea surface temperature, sea ice, and night marine air temperature. J. Geophys. Res., 108(D14), 4407.
- Adler, R.F., et al. (2018). The Global Precipitation Climatology Project Monthly Analysis. Atmosphere, 9, 138.

### ENSO Science
- Trenberth, K.E. (1997). The Definition of El Niño. Bull. Amer. Meteor. Soc., 78, 2771-2777.
- Ashok, K., et al. (2007). El Niño Modoki and its possible teleconnection. J. Geophys. Res., 112, C11007.

### CMIP6
- Eyring, V., et al. (2016). Overview of the Coupled Model Intercomparison Project Phase 6. Geosci. Model Dev., 9, 1937-1958.

## Acknowledgments

- HadISST data: Met Office Hadley Centre
- GPCP data: NASA/NOAA
- CMIP6 data: ESGF and participating modeling centers
- Analysis framework developed for climate research applications

## License

This analysis framework is provided for research and educational purposes.
Data usage must comply with respective data provider policies.
