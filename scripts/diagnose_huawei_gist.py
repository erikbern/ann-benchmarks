#!/usr/bin/env python3
"""Revalidate the persisted Huawei GIST tuned graph and screen adjacent query settings."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ann_benchmarks.runner import run_individual_query
from ann_benchmarks.algorithms.zvec_qg.module import digest


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def guard(a):
    require(os.environ.get('QG_ISOLATION_MANIFEST'), 'Run under the isolation wrapper')
    module('native_protocol',a.source/'tools/core/bench/qg_bench.py').check_quiet()
    require(os.sched_getaffinity(0)=={54},'CPU 54 required')
    for task in Path('/proc/self/task').iterdir():
        require(os.sched_getaffinity(int(task.name))=={54},'All retained threads must be pinned to CPU 54')


def thread_times():
    return {t.id:t.user_time+t.system_time for t in psutil.Process().threads()}


def measure(algo,data,params,out,a,rounds,expected=None):
    train,queries,truth,threshold=data
    guard(a)
    algo.set_query_arguments(params)
    run_individual_query(algo,train,queries,'euclidean',10,1,False)
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    for ri in range(rounds):
        guard(a)
        before=thread_times()
        attrs,results=run_individual_query(algo,train,queries,'euclidean',10,1,False)
        after=thread_times()
        activity={tid:max(0,t-before.get(tid,0)) for tid,t in after.items()}
        require(all(t<=.02 for tid,t in activity.items() if tid!=os.getpid()),'Non-main thread consumed CPU during queries')
        ids=np.asarray([[j for j,d in r] for t,r in results],dtype='<i8')
        distances=np.asarray([[d for j,d in r] for t,r in results])
        times=np.asarray([t for t,r in results])
        require(ids.shape==(1000,10) and np.all((ids>=0)&(ids<len(train))),'Full result IDs')
        require(np.all(np.diff(np.sort(ids,axis=1),axis=1)>0),'Unique IDs')
        require(np.isfinite(times).all() and np.all(times>0),'Times')
        exact=np.empty_like(distances)
        for start in range(0,1000,100):
            x=train[ids[start:start+100]].astype(np.float64)
            q=queries[start:start+100].astype(np.float64)
            exact[start:start+100]=np.sqrt(np.sum((x-q[:,None,:])**2,axis=2))
        require(np.allclose(distances,exact,rtol=3e-5,atol=3e-6),'Independent distances')
        recall=float(np.mean(distances<=threshold[:,None]+.001))
        strict=float(np.mean(np.any(ids[:,:,None]==truth[:,None,:],axis=2)))
        h=hashlib.sha256(ids.tobytes()).hexdigest()
        if expected:require(h==expected,'Returned IDs differ from historical same graph/parameters')
        raw=out/(str(ri)+'.hdf5')
        with h5py.File(raw,'w') as f:
            for k,v in attrs.items():f.attrs[k]=v
            f.attrs.update(algo='qsgngt',dataset='gist-960-euclidean',parameters_json=json.dumps(params))
            f.create_dataset('neighbors',data=ids);f.create_dataset('distances',data=distances);f.create_dataset('times',data=times)
        records.append(dict(round=ri,qps=1/float(times.mean()),ann_recall=recall,strict_id_recall=strict,thread_cpu_seconds=activity,
                            independent_ann_recall=float(np.mean(exact<=threshold[:,None]+.001)),
                            ids_sha256=h,file=str(raw),sha256=digest(raw)))
    require(len({r['ids_sha256'] for r in records})==1,'Unstable queries')
    row=dict(parameters=params,rounds=records,qps=max(r['qps'] for r in records),ann_recall=records[0]['ann_recall'])
    save(out/'measurement.json',row)
    print('POINT',params,'recall',row['ann_recall'],'QPS',round(row['qps']),flush=True)
    return row


def worker(a):
    import ngtpy
    guard(a)
    plan=json.loads((a.output/'plan.json').read_text())
    for f in plan['fingerprints']:
        require(digest(f['path'])==f['sha256'],'Fingerprint changed: '+f['path'])
    with h5py.File(a.bench_root/'data/gist-960-euclidean.hdf5') as f:
        data=(f['train'][:],f['test'][:],f['neighbors'][:,:10],f['distances'][:,9])
    require(data[0].shape==(1000000,960) and data[1].shape==(1000,960),'Full dataset')
    QSG=module('ann_benchmarks.algorithms.qsg_ngt.module',a.bench_root/'ann_benchmarks/algorithms/qsg_ngt/module.py').QSG
    algo=QSG(*plan['definition']['arguments'])
    # The same explicit load path used by the previous full-query vendor validation.
    algo.index=ngtpy.QuantizedIndex(plan['index'],plan['definition']['arguments'][-1]['max_edge'],objects=data[0].ctypes.data)
    algo.index.set_with_distance(False)
    for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),{54})
    guard(a)
    stage='screen' if a.repeat is None else 'repeat-'+str(a.repeat)
    dest=a.output/stage;dest.mkdir(exist_ok=False)
    save(dest/'runtime.json',dict(pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),threads=psutil.Process().num_threads(),
                                 status=Path('/proc/self/status').read_text(),numa_maps=Path('/proc/self/numa_maps').read_text(),
                                 ngtpy=ngtpy.__file__,ngtpy_sha256=digest(ngtpy.__file__)))
    params=plan['screen_points'] if a.repeat is None else json.loads((a.output/'selected.json').read_text())
    records=[]
    for i,p in enumerate(params):
        old=next((r for r in plan['historical'] if r['parameters']==p),None)
        row=measure(algo,data,p,dest/str(i),a,1 if a.repeat is None else 5,old['ids_sha256'] if old else None)
        records.append(row);save(dest/'measurements.json',records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench-root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--repeat',type=int)
    a=p.parse_args()
    if a.worker:return worker(a)
    guard(a)
    def run(extra,log):
        cmd=[sys.executable,str(Path(__file__).resolve()),'--bench-root',str(a.bench_root),'--source',str(a.source),'--output',str(a.output),'--worker',*extra]
        with (a.output/log).open('w') as stream:subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
    run([],'screen.log')
    rows=json.loads((a.output/'screen/measurements.json').read_text())
    plan=json.loads((a.output/'plan.json').read_text())
    selected=[r['parameters'] for r in plan['historical'] if r['target'] in [.9,.95,.99]]
    for target in [.9,.95,.99]:
        eligible=[r for r in rows if r['ann_recall']>=target]
        if eligible:
            best=max(eligible,key=lambda r:r['qps'])['parameters']
            if best not in selected:selected.append(best)
    save(a.output/'selected.json',selected)
    repeats=[]
    for repeat in range(3):
        run(['--repeat',str(repeat)],'repeat-'+str(repeat)+'.log')
        repeats.append(json.loads((a.output/('repeat-'+str(repeat))/'measurements.json').read_text()))
    summary=[]
    for i,params in enumerate(selected):
        rows=[r[i] for r in repeats]
        require(len({x['ids_sha256'] for r in rows for x in r['rounds']})==1,'Unstable independent repetitions')
        qps=[r['qps'] for r in rows]
        summary.append(dict(parameters=params,ann_recall=rows[0]['ann_recall'],strict_id_recall=rows[0]['rounds'][0]['strict_id_recall'],
                            repeat_best5_qps=qps,qps=float(np.median(qps))))
    save(a.output/'summary.json',summary)
    winners=[]
    for t in [.9,.95,.99]:
        eligible=[r for r in summary if r['ann_recall']>=t]
        winners.append(dict(target=t,best=max(eligible,key=lambda r:r['qps']) if eligible else None))
    save(a.output/'winners.json',winners)
    print(json.dumps(winners,indent=2),flush=True)


if __name__=='__main__':main()
