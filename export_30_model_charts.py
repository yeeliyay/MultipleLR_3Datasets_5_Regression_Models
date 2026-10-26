"""Three datasets * five regression models * two metrics = 30 PNG charts.
The x-axis is only the five designated feature counts for each dataset.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
FEATURES={
    'appliances':[6,12,17,22,27],
    'parkinsons':[4,8,12,16,19],
    'student':[6,12,18,25,32],
}
MODELS={'ols':'OLS','ridge':'Ridge','lasso':'Lasso','elasticnet':'ElasticNet','bayesianridge':'Bayesian Ridge'}
METRICS={'rmse':'RMSE','r2':'R-squared'}
OUTPUT=ROOT/'line_charts_30'

def main():
    saved=[]
    for ds, ks in FEATURES.items():
        source=ROOT/'results'/f'{ds}.json'
        if not source.exists():
            raise FileNotFoundError(f'{source} missing: run python train_all.py')
        data=json.loads(source.read_text(encoding='utf-8'))
        if data.get('schema_version')!=2:
            raise ValueError(f'{ds}: old results; run python train_all.py')
        lookup={(r['model'],r['k']):r for r in data['metrics']}
        for method, title in MODELS.items():
            if not all((method,k) in lookup for k in ks):
                raise ValueError(f'{ds}/{method}: some requested feature counts missing')
            for metric, ylabel in METRICS.items():
                ys=[lookup[method,k][metric] for k in ks]
                fig,ax=plt.subplots(figsize=(9,5.5))
                ax.plot(ks,ys,'-o',linewidth=2.5,markersize=7)
                ax.set_xticks(ks)
                ax.set_xlabel('Number of Features (k)')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{ds.title()} | {title} | {ylabel}')
                ax.grid(True, alpha=.3)
                for k,y in zip(ks,ys):ax.annotate(f'{y:.4f}',(k,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
                ax.margins(y=.2)
                path=OUTPUT/ds/f'{ds}_{method}_{metric}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
                saved.append(path)
    print(f'Exported {len(saved)} PNG files into {OUTPUT}')

if __name__=='__main__':main()
