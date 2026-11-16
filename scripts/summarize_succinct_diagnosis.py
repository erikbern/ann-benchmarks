"""Validate and summarize the same-snapshot native-loop ablations."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/succinct-diagnosis-20261116'


def main():
    rows=[]
    for file in sorted((OUT/'measurements').glob('*.json')):
        rows.extend(json.loads(file.read_text()))
    groups={}
    for row in rows:
        ids=np.load(OUT/'measurements'/f'{row["profile"]}-{row["label"]}-{row["cycle"]}-{row["target"]}.npy')
        assert hashlib.sha256(ids.tobytes()).hexdigest()==row['ids_sha256']
        assert ids.shape==(row['query_count'],10)
        assert len(row['native_seconds'])==7 and min(row['native_seconds'])>0
        assert np.isclose(row['native_qps'],len(ids)/median(row['native_seconds']))
        groups.setdefault((row['profile'],row['target'],row['label']),[]).append(row)
    aggregates={}
    for key,points in groups.items():
        assert len(points)==2 and {r['cycle'] for r in points}=={0,1},key
        assert len({r['ids_sha256'] for r in points})==1,key
        r=dict(points[0])
        r.update(native_qps=median(p['native_qps'] for p in points),
                 native_qps_min=min(p['native_qps'] for p in points),
                 native_qps_max=max(p['native_qps'] for p in points))
        aggregates[key]=r
    summary=[]
    labels=['exploration','succinct','control','fixed','prefetch','both']
    for job in json.loads((OUT/'plan.json').read_text()):
        for target in [.9,.95,.99]:
            points={label:aggregates[(job['profile'],target,label)] for label in labels}
            assert len({json.dumps(p['parameters'],sort_keys=True) for p in points.values()})==1
            assert len({p['index'] for p in points.values()})==1
            control=points['control']
            row=dict(dataset=job['dataset'],target=target,
                     parameters=control['parameters'],index=control['index'])
            for label,p in points.items():
                row[label+'_native_qps']=p['native_qps']
                row[label+'_strict_id_recall']=p['strict_id_recall']
                row[label+'_vs_control_pct']=100*(p['native_qps']/control['native_qps']-1)
                row[label+'_ids_equal_control']=p['ids_sha256']==control['ids_sha256']
                row[label+'_repeat_spread_pct']=100*(p['native_qps_max']/p['native_qps_min']-1)
            row['succinct_vs_exploration_pct']=100*(points['succinct']['native_qps']/points['exploration']['native_qps']-1)
            summary.append(row)
    with (OUT/'ablation-summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n')
        writer.writeheader()
        for row in summary:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,dict) else v for k,v in row.items()})
    (OUT/'ablation-summary.json').write_text(json.dumps(summary,indent=2))
    for row in summary:
        print(row['dataset'],row['target'],'branch',round(row['succinct_vs_exploration_pct'],2),
              'control',round(row['succinct_vs_control_pct'],2),
              'fixed',round(row['fixed_vs_control_pct'],2),
              'prefetch',round(row['prefetch_vs_control_pct'],2),
              'both',round(row['both_vs_control_pct'],2),
              'ids',[label for label in labels if not row[label+'_ids_equal_control']])


if __name__=='__main__':
    main()
