"""Three independently trained datasets; five real regression estimators.
Run `python train_all.py` locally to generate results/*.json before Render deploy.
Feature selection: F-test ranking fitted on fit partition (not test data).
Intervals: split-conformal prediction intervals using separate calibration partition.
"""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
RESULTS_DIR = BASE / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
CACHE = {}
DATASETS = {
    'appliances': {'title':'Appliances Energy Prediction','filename':'energydata_complete.csv',
      'target':'Appliances','exclude':['date'],'split':'time',
      'kaggle':'https://www.kaggle.com/competitions/machine-learning-sb1',
      'task':'以室內外溫溼度與環境資料預測家電用電量。',
      'note':'28 個候選特徵包含 rv1、rv2 兩個隨機對照特徵；僅作排名參照。'},
    'parkinsons': {'title':'Parkinsons Telemonitoring','filename':'parkinsons_updrs.data',
      'target':'total_UPDRS','exclude':['subject#','motor_UPDRS'],'split':'group',
      'kaggle':'https://www.kaggle.com/datasets/leilahasan/parkinson-dataset',
      'task':'依聲學特徵估計 total_UPDRS。',
      'note':'依受試者分組切割測試資料，排除 subject# 與 motor_UPDRS。'},
    'student': {'title':'Student Performance (Math)','filename':'student-mat.csv',
      'target':'G3','exclude':[],'split':'random',
      'kaggle':'https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set',
      'task':'依學生資訊及 G1/G2 成績預測 G3。',
      'note':'32 個原始特徵；類別經 one-hot 後，k 代表編碼後欄位數。G1/G2 是先前已知成績。'}
}
MODELS = {
    'ols': 'Multiple Linear Regression (OLS)',
    'ridge': 'Ridge Regression (L2)',
    'lasso': 'Lasso Regression (L1)',
    'elasticnet': 'Elastic Net Regression',
    'bayesianridge': 'Bayesian Ridge Regression',
}

def estimator(name):
    if name == 'ols': return LinearRegression()
    if name == 'ridge': return Ridge(alpha=1.0)
    if name == 'lasso': return Lasso(alpha=0.1, max_iter=12000, random_state=42)
    if name == 'elasticnet': return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=12000, random_state=42)
    if name == 'bayesianridge': return BayesianRidge()
    raise ValueError(f'Unknown model: {name}')

def result_path(key): return RESULTS_DIR / f'{key}.json'

def load_or_train(key, force=False):
    if key not in DATASETS: raise ValueError('Unknown dataset')
    if not force and key in CACHE: return CACHE[key]
    path = result_path(key)
    if not force and path.exists():
        with path.open(encoding='utf-8') as handle: result = json.load(handle)
        if result.get('schema_version') == 2:
            CACHE[key] = result
            return result
    result = analyze(key)
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as handle: json.dump(result, handle, ensure_ascii=False, allow_nan=False)
    tmp.replace(path)
    CACHE[key] = result
    return result

def load_data(key):
    cfg = DATASETS[key]
    path = BASE / 'data' / cfg['filename']
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run python download_data.py locally, then python train_all.py, then push results/*.json to GitHub.')
    df = pd.read_csv(path, sep=';' if key == 'student' else ',')
    df[cfg['target']] = pd.to_numeric(df[cfg['target']], errors='coerce')
    df = df.dropna(subset=[cfg['target']]).copy()
    if key == 'appliances': df = df.sort_values('date')
    return df

def split_indices(df, cfg):
    idx = np.arange(len(df))
    if cfg['split'] == 'time':
        n = int(len(idx)*.8)
        train, test = idx[:n], idx[n:]
        n_fit = int(len(train)*.8)
        fit, calibration = train[:n_fit], train[n_fit:]
    elif cfg['split'] == 'group':
        train, test = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(idx,groups=df['subject#']))
        fit_relative, cal_relative = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=24).split(train,groups=df.iloc[train]['subject#']))
        fit, calibration = train[fit_relative], train[cal_relative]
    else:
        train, test = train_test_split(idx, test_size=.2, random_state=42)
        fit, calibration = train_test_split(train, test_size=.2, random_state=24)
    return fit, calibration, test

def analyze(key):
    cfg=DATASETS[key]; df=load_data(key)
    xcols=[col for col in df.columns if col not in [cfg['target'], *cfg['exclude']]]
    Xraw=df[xcols]
    y=df[cfg['target']].to_numpy(dtype=float)
    fit_ids, cal_ids, test_ids = split_indices(df, cfg)
    nums=Xraw.select_dtypes(include='number').columns.tolist()
    cats=[col for col in xcols if col not in nums]
    stages=[]
    if nums: stages.append(('num', Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums))
    if cats: stages.append(('cat', Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats))
    pre = ColumnTransformer(stages, sparse_threshold=0)
    Xfit=pre.fit_transform(Xraw.iloc[fit_ids]); Xcal=pre.transform(Xraw.iloc[cal_ids]); Xtest=pre.transform(Xraw.iloc[test_ids])
    yf=y[fit_ids]; yc=y[cal_ids]; yt=y[test_ids]
    names=pre.get_feature_names_out().tolist()
    with np.errstate(all='ignore'):
        scores=f_regression(Xfit,yf)[0]
    scores=np.nan_to_num(scores,nan=0.0,posinf=1e12,neginf=0.0)
    rank=np.argsort(-scores,kind='stable')
    maxk=min(len(xcols), Xfit.shape[1], len(fit_ids)-2)
    if maxk < 1: raise ValueError('Not enough training rows or features')
    metrics=[]; plots={}; coefficients={}
    for model in MODELS:
        for k in range(1,maxk+1):
            selected=rank[:k]
            fit_X=Xfit[:,selected]; cal_X=Xcal[:,selected]; test_X=Xtest[:,selected]
            reg=estimator(model)
            reg.fit(fit_X,yf)
            cal_prediction=reg.predict(cal_X)
            test_prediction=reg.predict(test_X)
            # Split-conformal marginal prediction interval, alpha=.05.
            cal_errors=np.abs(yc-cal_prediction)
            qlevel=min(1.0, np.ceil((len(cal_errors)+1)*.95)/len(cal_errors))
            q=float(np.quantile(cal_errors,qlevel,method='higher'))
            rmse=float(np.sqrt(mean_squared_error(yt,test_prediction)))
            r2=float(r2_score(yt,test_prediction))
            mae=float(mean_absolute_error(yt,test_prediction))
            metrics.append({'model':model,'method':model,'k':k,'rmse':rmse,'r2':r2,'mae':mae,
                            'features':[names[i] for i in selected]})
            order=np.argsort(test_prediction)
            sampled=order[np.linspace(0,len(order)-1,min(150,len(order))).round().astype(int)]
            plots[f'{model}_{k}']={
                'actual':yt[sampled].tolist(),
                'predicted':test_prediction[sampled].tolist(),
                'lower':(test_prediction[sampled]-q).tolist(),
                'upper':(test_prediction[sampled]+q).tolist(),
                'residual':(yt[sampled]-test_prediction[sampled]).tolist(),
                'pi_coverage':float(np.mean(np.abs(yt-test_prediction)<=q)),
                'pi_mean_width':2*q,
            }
            coefficients[f'{model}_{k}']={
                'intercept':float(reg.intercept_),
                'coefs':[{'feature':names[i],'beta':float(b)} for i,b in zip(selected,reg.coef_)]
            }
    # Display first model default; do not label test-set-picked minimum as unbiased optimum.
    return {
        'schema_version':2,'key':key,'dataset':cfg,'count':len(df),'original_features':len(xcols),
        'encoded_features':len(names),'train':len(fit_ids),'calibration':len(cal_ids),'test':len(test_ids),
        'metrics':metrics,'plots':plots,'coefficients':coefficients,
        'feature_selection':'F-test ranking using fit split only',
        'target_stats':{'mean':float(np.mean(y)),'std':float(np.std(y)),'min':float(np.min(y)),'max':float(np.max(y))},
        'missing':int(df.isna().sum().sum()),
        'interval_note':'Split-conformal 95% prediction interval; marginal coverage assumes exchangeability of calibration and test observations. Time-series drift may invalidate that assumption.'
    }

@app.route('/')
def home(): return render_template('index.html',datasets=DATASETS)

@app.route('/api/analyze/<key>')
def get_analysis(key):
    if key not in DATASETS: return jsonify(error='Unknown dataset'),404
    try: return jsonify(load_or_train(key))
    except Exception as exc: return jsonify(error=f'{type(exc).__name__}: {exc}'),400

@app.route('/api/refresh/<key>', methods=['POST'])
def refresh(key):
    if key not in DATASETS: return jsonify(error='Unknown dataset'),404
    try: return jsonify(load_or_train(key,force=True))
    except Exception as exc: return jsonify(error=f'{type(exc).__name__}: {exc}'),400

if __name__=='__main__': app.run(host='127.0.0.1',port=int(os.getenv('PORT','5000')),debug=False)
