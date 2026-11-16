"""Isolated, same-binary GloVe timing, parameter and graph diagnostics.

Results are local experiments, not evidence of the unknown c9i timer or CPU.
The coordinator drains the current campaign worker and resumes its controller
in finally. The main campaign's immutable binaries and results are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/glove-gap-20261116'
CAMPAIGN = ROOT / 'results/comprehensive-icelake-20261116'
PRIOR = ROOT / 'results/succinct-diagnosis-20261116'


def save(name, data):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def phase(name, **data):
    record = dict(phase=name, time=time.time(), **data)
    save('progress.json', record)
    print(json.dumps(record), flush=True)


def worker():
    import numpy as np
    import yaml
    import zvec
    from py4bench import CoreSearcher
    from ann_benchmarks.algorithms.base.module import BaseANN
    from ann_benchmarks.distance import metrics
    from ann_benchmarks.runner import run_individual_query
    from comprehensive_icelake_worker import load_data, large_mappings
    os.sched_setaffinity(0, {48})
    overlay = PRIOR / 'builds/compact/overlay'
    assert Path(zvec.__file__).is_relative_to(overlay)
    binary = next((overlay/'zvec').glob('_zvec*.so'))
    job = next(j for j in json.loads((PRIOR/'plan.json').read_text())
               if j['dataset'] == 'glove-100-angular')
    source = Path.home()/'workspace/zvec/quantized_graph_exploration/zvec'
    config = yaml.safe_load((source/'tools/core/bench/config/glove_fp32_recall_bench.yaml').read_text())
    index_params = json.loads(config['IndexCommon']['IndexConfig'])
    train, queries, truth, thresholds, metric = load_data(job['dataset'])
    distance = metrics[metric].distance
    save('method.json', dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        index=job['index'], index_params=index_params, cpu=48, query_count=len(queries),
        selected_parameters=job['points'], source='Validated compact candidate, equivalent to merged 2387e019',
        note='Same full public query set for tuning and reporting; no claim of held-out generalization.'))

    def evaluate(ids):
        assert ids.shape == truth.shape
        assert np.all((ids >= 0) & (ids < len(train)))
        distances = np.asarray([[distance(q, train[int(i)]) for i in row] for q, row in zip(queries, ids)])
        return dict(recall=int(np.count_nonzero(distances <= thresholds[:, None]))/ids.size,
                    strict_id_recall=sum(len(set(a)&set(b)) for a,b in zip(ids,truth))/ids.size,
                    ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())

    def timed_python(searcher, mode):
        # Keep every returned list alive in both modes, matching the official
        # runner. All distance calculations are outside the query timer.
        total = 0.
        answers = []
        start_all = time.perf_counter()
        for q in queries:
            start = time.time()
            ids = searcher.search_list(q, 10)
            total += time.time()-start
            if mode == 'interleaved':
                assert len(ids) == len(set(ids))
                answers.append([(int(i), float(distance(q, train[i]))) for i in ids])
            else:
                answers.append(ids)
        wall = time.perf_counter()-start_all
        return dict(query_seconds=total, wall_seconds=wall)

    protocols = []
    modes = ['native', 'python', 'interleaved', 'official']
    for point in job['points']:
        searcher = CoreSearcher(job['index'], index_params, point['parameters'])
        ids = np.asarray([searcher.search_list(q,10) for q in queries], dtype=np.int64)
        correctness = evaluate(ids)
        algo = BaseANN()
        algo.name = 'glove-timing-diagnostic'
        algo.query = searcher.search_list
        save('mappings.json', large_mappings())
        for cycle, order in enumerate([modes, list(reversed(modes))]):
            for mode in order:
                phase('protocol', target=point['target'], cycle=cycle, mode=mode)
                if mode == 'native':
                    times = [searcher.time_native(queries,10) for _ in range(5)]
                    record = dict(seconds=times, qps_median=len(queries)/float(np.median(times)),
                                  qps_best=len(queries)/min(times))
                elif mode == 'official':
                    attrs, answers = run_individual_query(algo,train,queries,metric,10,5,False)
                    result_ids = np.asarray([[i for i,d in row] for _,row in answers],dtype=np.int64)
                    assert np.array_equal(ids,result_ids)
                    record = dict(qps_best=1/attrs['best_search_time'],
                                  last_query_seconds=sum(t for t,row in answers))
                else:
                    runs = [timed_python(searcher,mode) for _ in range(5)]
                    times = [r['query_seconds'] for r in runs]
                    record = dict(runs=runs, qps_best=len(queries)/min(times),
                                  qps_median=len(queries)/float(np.median(times)))
                protocols.append(dict(target=point['target'],cycle=cycle,mode=mode,
                                      **record,**correctness))
                save('protocols.json',protocols)
        del algo, searcher

    # Refine each existing winner's stopping margin, holding every other
    # setting and the full graph fixed. This isolates over-search from code.
    candidates = []
    margins = {0.9:[.04,.045,.048,.05,.052,.055],
               0.95:[.07,.075,.08,.082,.085,.087,.09],
               0.99:[.115,.12,.125,.13,.132,.135,.137,.14]}
    for point in job['points']:
        for margin in margins[point['target']]:
            params = dict(point['parameters'],quantized_graph_margin=margin)
            phase('margin_sweep',target=point['target'],margin=margin)
            searcher = CoreSearcher(job['index'],index_params,params)
            ids = np.asarray([searcher.search_list(q,10) for q in queries],dtype=np.int64)
            times = [searcher.time_native(queries,10) for _ in range(3)]
            candidates.append(dict(target=point['target'],parameters=params,
                native_qps=len(queries)/float(np.median(times)),native_seconds=times,**evaluate(ids)))
            save('margin-sweep.json',candidates)
            del searcher

    selected = []
    for target in [.9,.95,.99]:
        eligible = [r for r in candidates if r['target']==target and r['recall']>=target]
        winners = sorted(eligible,key=lambda r:r['native_qps'],reverse=True)[:2]
        assert winners
        base = next(p for p in job['points'] if p['target']==target)
        for params in [base['parameters']]+[r['parameters'] for r in winners]:
            item = dict(target=target,parameters=params)
            if item not in selected:
                selected.append(item)
    save('selected.json',selected)
    confirmations = []
    for cycle, points in enumerate([selected,list(reversed(selected))]):
        for point in points:
            phase('official_confirmation',cycle=cycle,**point)
            searcher = CoreSearcher(job['index'],index_params,point['parameters'])
            for q in queries:
                searcher.search_list(q,10)
            algo = BaseANN()
            algo.name = 'glove-margin-confirmation'
            algo.query = searcher.search_list
            attrs,answers = run_individual_query(algo,train,queries,metric,10,5,False)
            ids = np.asarray([[i for i,d in row] for _,row in answers],dtype=np.int64)
            confirmations.append(dict(**point,cycle=cycle,qps=1/attrs['best_search_time'],**evaluate(ids)))
            save('confirmations.json',confirmations)
            del algo,searcher

    # Different builds of the graph, with the same code and query parameters.
    graph_results=[]
    indexes=[job['index'],str(CAMPAIGN/'indexes/glove-succinct-v2/index')]
    for cycle,order in enumerate([indexes,list(reversed(indexes))]):
        for index in order:
            for point in job['points']:
                phase('graph_comparison',cycle=cycle,index=index,target=point['target'])
                searcher=CoreSearcher(index,index_params,point['parameters'])
                ids=np.asarray([searcher.search_list(q,10) for q in queries],dtype=np.int64)
                times=[searcher.time_native(queries,10) for _ in range(5)]
                graph_results.append(dict(index=index,cycle=cycle,**point,
                    native_qps=len(queries)/float(np.median(times)),**evaluate(ids)))
                save('graphs.json',graph_results)
                del searcher
    phase('building_diagnostic_profile')
    subprocess.run([sys.executable,str(ROOT/'scripts/profile_glove_stages.py')],check=True)
    phase('worker_finished')


def coordinate():
    import psutil
    from diagnose_succinct import environment
    assert not (OUT/'controller.json').exists(), 'Use a new output directory for a new run'
    state=json.loads((CAMPAIGN/'controller.json').read_text())
    controller=psutil.Process(state['pid'])
    assert abs(controller.create_time()-state['created'])<.01
    assert controller.status()!=psutil.STATUS_STOPPED
    assert any('comprehensive_icelake.py' in s for s in controller.cmdline())
    save('controller.json',state)
    controller.send_signal(signal.SIGSTOP)
    try:
        while any(p.status()!=psutil.STATUS_ZOMBIE for p in controller.children(recursive=True)):
            phase('waiting_for_current_campaign_worker')
            time.sleep(5)
        # Existing full-scan dispatchers are still held by the main controller.
        for entry in json.loads((CAMPAIGN/'paused-dispatchers.json').read_text()):
            p=psutil.Process(entry['pid'])
            assert abs(p.create_time()-entry['created'])<.01 and p.status()==psutil.STATUS_STOPPED
        phase('isolated_worker')
        subprocess.run([sys.executable,__file__,'--worker'],env=environment('compact'),check=True)
    finally:
        if controller.is_running() and abs(controller.create_time()-state['created'])<.01:
            controller.send_signal(signal.SIGCONT)
            save('controller-resumed.json',dict(**state,time=time.time()))
    phase('finished')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    worker() if args.worker else coordinate()
