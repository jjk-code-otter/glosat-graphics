from pathlib import Path
import requests
import shutil


def fetch(url: str, outdir: Path, filename: str) -> None:
    """
    Fetcher for a standard URL.

    Parameters
    ----------
    url: str
        URL of the file to be downloaded.
    outdir: Path
        Path of the directory to which the output will be written
    filename: str
        Filename to save file as locally

    Returns
    -------
    None
    """
    out_path = outdir / filename

    try:
        r = requests.get(url, stream=True, headers={'User-agent': 'Mozilla/5.0'})

        if r.status_code == 200:
            with open(out_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)

        if r.status_code == 403:
            print("URL not retrievable automatically. Try downloading manually")
            print(url)

    except requests.exceptions.ConnectionError:
        print(f"Couldn't connect to {url}")


url_list = [
    ['https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1029%2F2006GL026152&file=grl21511-sup-0002-ts01.txt',
     'grl21511-sup-0002-ts01.txt'],
    ['https://gaw.kishou.go.jp/static/publications/global_mean_mole_fractions/2023/co2_annual_20231115.csv',
     'co2_annual_20231115.csv'],
    ['https://gaw.kishou.go.jp/static/publications/global_mean_mole_fractions/2023/ch4_annual_20231115.csv',
     'ch4_annual_20231115.csv'],
    ['https://gaw.kishou.go.jp/static/publications/global_mean_mole_fractions/2023/n2o_annual_20231115.csv',
     'n2o_annual_20231115.csv'],
    ['https://dap.ceda.ac.uk/badc/ar6_wg1/data/ch_03/ch3_fig09/v20211028/panel_b/fig_3_9_b.nc',
     'fig_3_9_b.nc'],
    ['https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz',
     'HadISST_sst.nc.gz'],
    ['https://jjk-code-otter.github.io/demo-dash/Dashboard2023/formatted_data/Global_temperature_data_files.zip',
     'Global_temperature_data_files.zip'],
    ['https://crudata.uea.ac.uk/cru/data/nao/nao_3dp.dat',
     'nao_3dp.dat.txt.txt'],
    ['https://www.ncei.noaa.gov/pub/data/paleo/contributions_by_author/abram2014/abram2014sam-noaa.txt',
     'abram2014sam-noaa.txt'],
    ['http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.seas.txt',
     'newsam.1957.2007.seas.txt']
]

out_dir = Path('InputData')
out_dir.mkdir(exist_ok=True)

for url in url_list:
    filename = url[1]
    fetch(url[0], out_dir, filename)
