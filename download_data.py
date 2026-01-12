#!/usr/bin/env python3
"""
ENSO Analysis Data Download Script

Downloads real observational and CMIP6 model data for ENSO analysis.

Data Sources:
1. HadISST - Sea Surface Temperature from Met Office Hadley Centre
2. GPCP - Global Precipitation Climatology Project v2.3
3. CMIP6 models from ESGF

Run this script to download all required data before running the analysis.
"""

import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA SOURCE URLS AND INSTRUCTIONS
# =============================================================================

DATA_SOURCES = {
    'HadISST': {
        'description': 'Hadley Centre Sea Ice and Sea Surface Temperature (HadISST1)',
        'url_netcdf': 'https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz',
        'url_download_page': 'https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html',
        'erddap_url': 'https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdHadISST.nc',
        'psl_thredds': 'https://psl.noaa.gov/thredds/dodsC/Datasets/hadisst/HadISST_sst.nc',
        'resolution': '1° x 1°',
        'period': '1870-present',
        'variables': ['sst'],
        'reference': 'Rayner et al. (2003) J. Geophys. Res., 108(D14), 4407'
    },
    'ERSST_v5': {
        'description': 'NOAA Extended Reconstructed SST Version 5 (alternative to HadISST)',
        'url_netcdf': 'https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc',
        'psl_thredds': 'https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc',
        'resolution': '2° x 2°',
        'period': '1854-present',
        'variables': ['sst'],
        'reference': 'Huang et al. (2017) J. Climate, 30, 8179-8205'
    },
    'GPCP_monthly': {
        'description': 'Global Precipitation Climatology Project Monthly v2.3',
        'url_netcdf': 'https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc',
        'ncei_url': 'https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-monthly/',
        'psl_thredds': 'https://psl.noaa.gov/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc',
        'resolution': '2.5° x 2.5°',
        'period': '1979-present',
        'variables': ['precip'],
        'reference': 'Adler et al. (2018) Atmosphere, 9, 138'
    }
}

CMIP6_MODELS = {
    'CESM2': {
        'institution': 'NCAR (National Center for Atmospheric Research)',
        'country': 'USA',
        'esgf_source_id': 'CESM2'
    },
    'UKESM1-0-LL': {
        'institution': 'Met Office Hadley Centre',
        'country': 'UK',
        'esgf_source_id': 'UKESM1-0-LL'
    },
    'MPI-ESM1-2-HR': {
        'institution': 'Max Planck Institute for Meteorology',
        'country': 'Germany',
        'esgf_source_id': 'MPI-ESM1-2-HR'
    },
    'GFDL-ESM4': {
        'institution': 'NOAA Geophysical Fluid Dynamics Laboratory',
        'country': 'USA',
        'esgf_source_id': 'GFDL-ESM4'
    },
    'ACCESS-ESM1-5': {
        'institution': 'CSIRO & Bureau of Meteorology',
        'country': 'Australia',
        'esgf_source_id': 'ACCESS-ESM1-5'
    }
}


def print_download_instructions():
    """Print detailed instructions for downloading data."""
    print("=" * 80)
    print("ENSO ANALYSIS DATA DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("PART 1: OBSERVATIONAL DATA")
    print("=" * 80)
    
    print("\n--- HadISST (Primary SST Dataset) ---")
    print(f"Description: {DATA_SOURCES['HadISST']['description']}")
    print(f"Resolution: {DATA_SOURCES['HadISST']['resolution']}")
    print(f"Period: {DATA_SOURCES['HadISST']['period']}")
    print("\nDownload Options:")
    print(f"  1. Direct NetCDF: {DATA_SOURCES['HadISST']['url_netcdf']}")
    print(f"  2. ERDDAP (subset): {DATA_SOURCES['HadISST']['erddap_url']}")
    print(f"  3. Download page: {DATA_SOURCES['HadISST']['url_download_page']}")
    print("\nCommand:")
    print(f"  wget {DATA_SOURCES['HadISST']['url_netcdf']} -O data/HadISST_sst.nc.gz")
    print("  gunzip data/HadISST_sst.nc.gz")
    
    print("\n--- ERSST v5 (Alternative SST Dataset) ---")
    print(f"Description: {DATA_SOURCES['ERSST_v5']['description']}")
    print(f"Resolution: {DATA_SOURCES['ERSST_v5']['resolution']}")
    print("\nCommand:")
    print(f"  wget {DATA_SOURCES['ERSST_v5']['url_netcdf']} -O data/ersst_v5_sst.nc")
    
    print("\n--- GPCP Monthly Precipitation ---")
    print(f"Description: {DATA_SOURCES['GPCP_monthly']['description']}")
    print(f"Resolution: {DATA_SOURCES['GPCP_monthly']['resolution']}")
    print(f"Period: {DATA_SOURCES['GPCP_monthly']['period']}")
    print("\nCommand:")
    print(f"  wget {DATA_SOURCES['GPCP_monthly']['url_netcdf']} -O data/gpcp_precip.nc")
    
    print("\n" + "=" * 80)
    print("PART 2: CMIP6 MODEL DATA")
    print("=" * 80)
    print("\nDownload from ESGF (Earth System Grid Federation):")
    print("  Web Portal: https://esgf-node.llnl.gov/search/cmip6/")
    print("\nSearch Parameters:")
    print("  - Project: CMIP6")
    print("  - Experiment: historical, ssp126, ssp245, ssp585")
    print("  - Variable: tos (sea surface temperature), pr (precipitation)")
    print("  - Frequency: mon")
    print("  - Table: Omon (ocean monthly), Amon (atmosphere monthly)")
    
    print("\nModels to download:")
    for model, info in CMIP6_MODELS.items():
        print(f"  - {model} ({info['institution']}, {info['country']})")
    
    print("\nExample ESGF Search URL for CESM2 historical SST:")
    print("  https://esgf-node.llnl.gov/search/cmip6/?source_id=CESM2&experiment_id=historical&variable_id=tos&frequency=mon")
    
    print("\n" + "=" * 80)
    print("ALTERNATIVE: Using Python to Download via OPeNDAP")
    print("=" * 80)
    print("""
import xarray as xr

# HadISST via PSL THREDDS
hadisst = xr.open_dataset(
    'https://psl.noaa.gov/thredds/dodsC/Datasets/hadisst/HadISST_sst.nc',
    engine='netcdf4'
)
hadisst.to_netcdf('data/HadISST_sst.nc')

# GPCP via PSL THREDDS  
gpcp = xr.open_dataset(
    'https://psl.noaa.gov/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc',
    engine='netcdf4'
)
gpcp.to_netcdf('data/gpcp_precip.nc')
    """)
    
    print("\n" + "=" * 80)
    print("EXPECTED FILE STRUCTURE")
    print("=" * 80)
    print("""
data/
├── observations/
│   ├── HadISST_sst.nc           # SST observations
│   └── gpcp_precip.nc           # Precipitation observations
├── cmip6/
│   ├── CESM2/
│   │   ├── tos_Omon_CESM2_historical_*.nc
│   │   ├── tos_Omon_CESM2_ssp126_*.nc
│   │   ├── tos_Omon_CESM2_ssp245_*.nc
│   │   └── tos_Omon_CESM2_ssp585_*.nc
│   ├── UKESM1-0-LL/
│   │   └── ...
│   ├── MPI-ESM1-2-HR/
│   │   └── ...
│   ├── GFDL-ESM4/
│   │   └── ...
│   └── ACCESS-ESM1-5/
│       └── ...
    """)


def check_data_availability():
    """Check which data files are already downloaded."""
    print("\nChecking data availability...")
    
    expected_files = {
        'HadISST': DATA_DIR / 'observations' / 'HadISST_sst.nc',
        'GPCP': DATA_DIR / 'observations' / 'gpcp_precip.nc',
    }
    
    for name, path in expected_files.items():
        if path.exists():
            print(f"  ✓ {name}: Found at {path}")
        else:
            print(f"  ✗ {name}: Not found at {path}")
    
    # Check CMIP6 models
    cmip6_dir = DATA_DIR / 'cmip6'
    for model in CMIP6_MODELS.keys():
        model_dir = cmip6_dir / model
        if model_dir.exists() and any(model_dir.glob('*.nc')):
            files = list(model_dir.glob('*.nc'))
            print(f"  ✓ {model}: Found {len(files)} files")
        else:
            print(f"  ✗ {model}: No data found")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        check_data_availability()
    else:
        print_download_instructions()
        print("\n" + "=" * 80)
        check_data_availability()
        print("\n" + "=" * 80)
        print("Run 'python download_data.py --check' to verify data availability")
        print("=" * 80)
