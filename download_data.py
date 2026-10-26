"""Download original data from the original UCI repositories; Kaggle mirrors are cited in README.
To use strictly Kaggle downloads, put their identically named CSV files in data/ instead.
"""
from pathlib import Path
from io import BytesIO
import zipfile,requests
D=Path(__file__).resolve().parent/'data';D.mkdir(exist_ok=True)
files=[
 ('appliances','https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip','energydata_complete.csv'),
 ('parkinsons','https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip','parkinsons_updrs.data'),
 ('student','https://archive.ics.uci.edu/static/public/320/student+performance.zip','student-mat.csv')]
def extract_from_zip(blob, filename, level=0):
    """Search outer and nested ZIP files (UCI student data contains student.zip)."""
    if level > 4:
        raise FileNotFoundError(f'{filename}: nested ZIP depth exceeded')
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        members = archive.namelist()
        matches = [entry for entry in members if entry.replace('\\', '/').split('/')[-1].lower() == filename.lower()]
        if matches:
            return archive.read(matches[0])
        for entry in members:
            if entry.lower().endswith('.zip'):
                try:
                    return extract_from_zip(archive.read(entry), filename, level + 1)
                except (FileNotFoundError, zipfile.BadZipFile):
                    continue
        raise FileNotFoundError(f'{filename} not in archive; found {members[:15]}')

for key, url, name in files:
    if (D / name).exists():
        print('Exists:', name)
        continue
    print('Downloading:', key)
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        data = extract_from_zip(r.content, name)
        (D / name).write_bytes(data)
        print('Saved:', D / name)
    except Exception as e:
        print('FAILED:', e, '\nPlease download the dataset from the Kaggle link in README and save', name, 'into data/')
