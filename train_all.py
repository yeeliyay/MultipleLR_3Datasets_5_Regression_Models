"""Train and cache the three datasets independently with five regressors."""
from app import DATASETS, MODELS, load_or_train, result_path

if __name__ == '__main__':
    for key, cfg in DATASETS.items():
        print(f'\n=== {cfg["title"]} ===', flush=True)
        try:
            r = load_or_train(key, force=True)
            print(f'Saved: {result_path(key)}',flush=True)
            print(f'{len(MODELS)} regression models x {max(a["k"] for a in r["metrics"])} k values = {len(r["metrics"])} rows',flush=True)
        except Exception as exc:
            print(f'FAILED {key}: {type(exc).__name__}: {exc}',flush=True)
