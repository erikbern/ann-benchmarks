#!/usr/bin/env python3
"""Verify every Huawei diagnostic raw round and the final median-of-best-five winners."""
import argparse
import json
from pathlib import Path
import sys
import hashlib

import h5py
import numpy as np
import docker
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ann_benchmarks.algorithms.zvec_qg.module import digest


def require(ok,msg):
    if not ok:raise ValueError(msg)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();root=a.run
    plan=json.loads((root/'plan.json').read_text())
    require(digest(ROOT/'scripts/diagnose_huawei_gist.py')==plan['code_sha256'],'Measured code changed')
    with h5py.File(next(f['path'] for f in plan['fingerprints'] if f['path'].endswith('gist-960-euclidean.hdf5'))) as f:
        thresholds=f['distances'][:,9];truth=f['neighbors'][:,:10]
    rounds=0;nonmain=0;queries=0
    manifests=[root/'screen/measurements.json']+sorted(root.glob('repeat-*/measurements.json'))
    require(len(manifests)==4,'Missing stage')
    for path in manifests:
        runtime=json.loads((path.parent/'runtime.json').read_text())
        require(runtime['affinity']==[54],'Affinity')
        for row in json.loads(path.read_text()):
            for r in row['rounds']:
                require(digest(r['file'])==r['sha256'],'Raw file changed')
                with h5py.File(r['file']) as f:
                    ids=f['neighbors'][:];dist=f['distances'][:];times=f['times'][:]
                require(ids.shape==(1000,10) and np.isfinite(times).all() and (times>0).all(),'Full query set')
                require(hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()==r['ids_sha256'],'IDs hash')
                require(float(np.mean(dist<=thresholds[:,None]+.001))==r['ann_recall'],'ANN recall')
                require(float(np.mean(np.any(ids[:,:,None]==truth[:,None,:],axis=2)))==r['strict_id_recall'],'Strict recall')
                require(np.isclose(1/times.mean(),r['qps'],rtol=1e-12),'QPS')
                extra=sum(t for tid,t in r['thread_cpu_seconds'].items() if int(tid)!=runtime['pid'])
                nonmain+=extra;rounds+=1;queries+=len(ids)
    stages=[json.loads((root/f'repeat-{i}/measurements.json').read_text()) for i in range(3)]
    summary=json.loads((root/'summary.json').read_text())
    for i,s in enumerate(summary):
        reps=[max(r['qps'] for r in stage[i]['rounds']) for stage in stages]
        require(reps==s['repeat_best5_qps'] and float(np.median(reps))==s['qps'],'Median aggregation')
    isolation=json.loads((root/'isolation.json').read_text())
    require(isolation.get('restored_at') and not isolation['restore_errors'],'Isolation restore failed')
    states=[];client=docker.from_env()
    for entry in isolation['containers']:
        try:c=client.containers.get(entry['id']);c.reload();state=c.status
        except docker.errors.NotFound:state='completed/removed'
        require(state!='paused','Container still paused');states.append(dict(id=entry['id'],state=state))
    for entry in isolation['processes']:
        try:
            p=psutil.Process(entry['pid']);state=p.status() if p.create_time()==entry['created'] else 'replaced'
        except psutil.NoSuchProcess:state='completed'
        require(state!='stopped','Controller still stopped');states.append(dict(pid=entry['pid'],state=state))
    report=dict(passed=True,screen_points=33,formal_settings=len(summary),formal_rounds=len(summary)*15,
                all_checked_rounds=rounds,timed_queries=queries,nonmain_cpu_seconds=nonmain,
                source_fingerprint_matches=True,raw_files_verified=True,qps_recomputed=True,restored=states)
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
