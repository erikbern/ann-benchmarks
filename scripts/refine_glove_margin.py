"""Resolve narrow recall-threshold brackets before final diagnostic profiling."""
import json
import os
from statistics import median

from diagnose_glove_gap import OUT, save, phase


def main():
    import numpy as np
    from py4bench import CoreSearcher
    from comprehensive_icelake_worker import load_data
    from ann_benchmarks.algorithms.base.module import BaseANN
    from ann_benchmarks.distance import metrics
    from ann_benchmarks.runner import run_individual_query
    import hashlib
    os.sched_setaffinity(0,{48})
    method=json.loads((OUT/'method.json').read_text())
    candidates=json.loads((OUT/'margin-sweep.json').read_text())
    confirmations=json.loads((OUT/'confirmations.json').read_text())
    assert not (OUT/'refined-boundaries.json').exists()
    train,queries,truth,thresholds,metric=load_data('glove-100-angular')

    def evaluate(ids):
        distances=np.asarray([[metrics[metric].distance(q,train[int(i)]) for i in row]
                              for q,row in zip(queries,ids)])
        return dict(recall=int(np.count_nonzero(distances<=thresholds[:,None]))/ids.size,
            strict_id_recall=sum(len(set(a)&set(b)) for a,b in zip(ids,truth))/ids.size,
            ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())

    new=[]
    selected=[]
    for target in [.9,.95,.99]:
        rows=[r for r in candidates if r['target']==target]
        below=max((r for r in rows if r['recall']<target),key=lambda r:r['parameters']['quantized_graph_margin'])
        above=min((r for r in rows if r['recall']>=target),key=lambda r:r['parameters']['quantized_graph_margin'])
        low=below['parameters']['quantized_graph_margin']
        high=above['parameters']['quantized_graph_margin']
        for step in range(3):
            margin=round((low+high)/2,8)
            params=dict(above['parameters'],quantized_graph_margin=margin)
            phase('refine_margin_boundary',target=target,margin=margin)
            searcher=CoreSearcher(method['index'],method['index_params'],params)
            ids=np.asarray([searcher.search_list(q,10) for q in queries],dtype=np.int64)
            times=[searcher.time_native(queries,10) for _ in range(3)]
            record=dict(target=target,parameters=params,native_seconds=times,
                native_qps=len(queries)/median(times),**evaluate(ids))
            new.append(record);rows.append(record);candidates.append(record)
            save('margin-sweep.json',candidates)
            if record['recall']>=target: high=margin
            else: low=margin
            del searcher
        best=max((r for r in rows if r['recall']>=target),key=lambda r:r['native_qps'])
        if not any(r['parameters']==best['parameters'] for r in confirmations):
            selected.append(dict(target=target,parameters=best['parameters']))
    save('refined-boundaries.json',new)
    for cycle,points in enumerate([selected,list(reversed(selected))]):
        for point in points:
            phase('refined_official_confirmation',cycle=cycle,**point)
            searcher=CoreSearcher(method['index'],method['index_params'],point['parameters'])
            for q in queries: searcher.search_list(q,10)
            algo=BaseANN();algo.name='glove-refined-margin';algo.query=searcher.search_list
            attrs,answers=run_individual_query(algo,train,queries,metric,10,5,False)
            ids=np.asarray([[i for i,d in row] for _,row in answers],dtype=np.int64)
            confirmations.append(dict(**point,cycle=cycle,qps=1/attrs['best_search_time'],**evaluate(ids)))
            save('confirmations.json',confirmations)
            del algo,searcher


def measure_affinity(method, queries):
    from py4bench import CoreSearcher
    # The imported plans use CPUs 10/12, historical docs CPU0, and the local
    # isolated campaign CPU48. Check whether selecting CPU48 explains the gap.
    affinity=[]
    for point in method['selected_parameters']:
        if point['target']==.95: continue
        searcher=CoreSearcher(method['index'],method['index_params'],point['parameters'])
        for position,cpu in enumerate([48,10,0,48]):
            os.sched_setaffinity(0,{cpu})
            phase('cpu_affinity_control',target=point['target'],cpu=cpu,position=position)
            searcher.time_native(queries,10)
            times=[searcher.time_native(queries,10) for _ in range(3)]
            affinity.append(dict(**point,cpu=cpu,position=position,native_seconds=times,
                                 native_qps=len(queries)/median(times)))
            save('cpu-affinity.json',affinity)
        del searcher
    os.sched_setaffinity(0,{48})


if __name__=='__main__': main()
