"""Same-snapshot source ablations; keep the main campaign binaries immutable."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/succinct-diagnosis-20261116'
CAMPAIGN = ROOT / 'results/comprehensive-icelake-20261116'
EXPLORATION = Path.home() / 'workspace/zvec/quantized_graph_exploration/zvec'
SUCCINCT = Path.home() / 'workspace/zvec/quantized_graph_exploration_succinct/zvec'
BUILD = SUCCINCT / 'build-ann'


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def environment(label):
    env = os.environ.copy()
    overlay = (CAMPAIGN / 'overlays' / f'{label}-portable' if label in ['exploration','succinct']
               else OUT / 'builds' / label / 'overlay')
    env.update(PYTHONPATH=f'{overlay}:{EXPLORATION / "py4bench/src"}:{ROOT}',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               ZVEC_TEST_SYMQG='1')
    return env


def rebuild(label, kernel=None, prefetch_override=False):
    if label in ['control','fixed','prefetch','both']:
        original_kernel=(SUCCINCT/'src/turbo/distance/avx512/symqg/symqg_kernels_avx512.cc').read_text()
        assert 'symqg_scan_impl' not in original_kernel, (
            'Rebuild ablation baselines from succinct a3f77830, not the optimized checkout; '
            'existing frozen binaries remain reusable')
    folder = OUT / 'builds' / label
    folder.mkdir(parents=True, exist_ok=True)
    include = folder / 'include'
    include.mkdir(exist_ok=True)
    for header in ['turbo/distance/common/symqg_distance.h',
                   'algorithm/quantized_graph/quantized_graph.h']:
        (include/header).unlink(missing_ok=True)
    if label in ['fixed','both']:
        header = 'turbo/distance/common/symqg_distance.h'
        original = (SUCCINCT / 'src' / header).read_text()
        reference = (EXPLORATION / 'src' / header).read_text()
        begin = reference.index('template <size_t Fixed>')
        end = reference.index('inline void ScanSymQGBatchUnrolled', begin)
        addition = reference[begin:end]
        changed = original.replace('#include <vector>', '#include <vector>\n#include <immintrin.h>')
        changed = changed.replace('inline void ScanSymQGBatch(', addition + 'inline void ScanSymQGBatch(', 1)
        assert changed.count('if (symqg_scan(') == 1
        changed = changed.replace('if (symqg_scan(', 'if (SymQGFixedScanDispatch(', 1)
        path = include / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(changed)
    if label in ['prefetch','both'] or prefetch_override:
        header = 'algorithm/quantized_graph/quantized_graph.h'
        original = (SUCCINCT / 'src/core' / header).read_text()
        changed = original.replace('        const bool changes_first = bins[i] <= beam.first();\n', '')
        changed = changed.replace('        if (changes_first) prefetch_next();\n', '')
        changed = changed.replace('    void prefetch_next() {\n',
            '    void prefetch_next() {\n'
            '      while (beam.has_next() && visited->visited(beam.next_id())) beam.pop();\n', 1)
        assert changed != original
        path = include / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(changed)
    commands = json.loads((OUT / 'original-build-commands.json').read_text())
    recorded = []
    replacements = {}
    for target, archive, subdir in [
        ('src/core/algorithm/hnsw/CMakeFiles/core_knn_hnsw_objects.dir/hnsw_quantized_graph.cc.o',
         'lib/libcore_knn_hnsw.a', 'hnsw'),
        ('src/core/CMakeFiles/zvec_core.dir/algorithm/hnsw/hnsw_quantized_graph.cc.o',
         'lib/libzvec_core.a', 'core')]:
        obj = folder / subdir / 'hnsw_quantized_graph.cc.o'
        obj.parent.mkdir(exist_ok=True)
        cmd = shlex.split(commands[target])
        cmd[1:1] = ['-I' + str(include), '-I' + str(SUCCINCT/'src/core/algorithm/quantized_graph')]
        for flag, value in [('-o', str(obj)), ('-MF', str(obj)+'.d'), ('-MT', str(obj))]:
            cmd[cmd.index(flag)+1] = value
        recorded.append(cmd)
        with (folder / (subdir+'.log')).open('w') as log:
            subprocess.run(['taskset','-c','50',*cmd], cwd=BUILD, stdout=log, stderr=subprocess.STDOUT, check=True)
        dependency_text = Path(str(obj)+'.d').read_text()
        for header in include.rglob('*.h'):
            assert str(header) in dependency_text, ('Override not compiled', header)
        copied = folder / Path(archive).name
        shutil.copy2(BUILD / archive, copied)
        subprocess.run(['ar','r',str(copied),str(obj)], check=True)
        replacements[archive] = str(copied)
    if kernel:
        target='src/turbo/CMakeFiles/zvec_turbo.dir/distance/avx512/symqg/symqg_kernels_avx512.cc.o'
        original=subprocess.check_output([str(ROOT/'.venv/bin/ninja'),'-t','commands',target],
                                         cwd=BUILD,text=True).splitlines()[-1]
        cmd=shlex.split(original)
        obj=folder/'symqg_kernels_avx512.cc.o'
        for flag,value in [('-c',str(kernel)),('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:
            cmd[cmd.index(flag)+1]=value
        recorded.append(cmd)
        with (folder/'kernel.log').open('w') as log:
            subprocess.run(['taskset','-c','50',*cmd],cwd=BUILD,stdout=log,stderr=subprocess.STDOUT,check=True)
        archive=folder/'libzvec_turbo.a'
        shutil.copy2(BUILD/'lib/libzvec_turbo.a',archive)
        subprocess.run(['ar','r',str(archive),str(obj)],check=True)
        replacements['lib/libzvec_turbo.a']=str(archive)
    name = 'lib/_zvec.cpython-310-x86_64-linux-gnu.so'
    # Ninja wraps the compiler command with ':' and strip; execute only the
    # compiler argv, preserving literal $ORIGIN without invoking a shell.
    link = shlex.split(commands[name].split('&&')[1].strip())
    overlay = folder / 'overlay/zvec'
    shutil.copytree(SUCCINCT / 'python/zvec', overlay, dirs_exist_ok=True)
    binary = overlay / Path(name).name
    link[link.index('-o')+1] = str(binary)
    link = [replacements.get(arg, arg) for arg in link]
    recorded.append(link)
    with (folder/'link.log').open('w') as log:
        subprocess.run(['taskset','-c','50',*link], cwd=BUILD, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['strip',str(binary)], check=True)
    save(folder/'commands.json', recorded)
    save(folder/'binary.json', dict(binary=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    with (folder/'tests.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pytest','-o','addopts=',str(EXPLORATION/'py4bench/tests'),'-q'],
                       cwd=ROOT,env=environment(label),stdout=log,stderr=subprocess.STDOUT,check=True)


def measure(label, profile, cycle, official=False):
    import h5py
    import numpy as np
    import yaml
    from py4bench import CoreSearcher
    import zvec
    os.sched_setaffinity(0,{48})
    plan = json.loads((OUT/'plan.json').read_text())
    job = next(j for j in plan if j['profile']==profile)
    with h5py.File(ROOT/'data'/(job['dataset']+'.hdf5')) as f:
        queries, truth = f['test'][:],f['neighbors'][:,:10]
    index_params = json.loads(yaml.safe_load((EXPLORATION / 'tools/core/bench/config' /
        (job['dataset'].split('-')[0]+'_fp32_recall_bench.yaml')).read_text())['IndexCommon']['IndexConfig'])
    if official:
        from comprehensive_icelake_worker import load_data, measure as ann_measure
        from ann_benchmarks.algorithms.base.module import BaseANN
        data=load_data(job['dataset'])
        for point in job['points']:
            searcher=CoreSearcher(job['index'],index_params,point['parameters'])
            algo=BaseANN()
            algo.name=f'compact-validation({label},{profile})'
            algo.query=searcher.search_list
            folder=OUT/'official'/profile/label/str(cycle)/str(point['target'])
            ann_measure(algo,data,point['parameters'],folder,dict(dataset=job['dataset'],
                variant=label,profile=profile,build='portable',phase='compact-official',
                cycle=cycle,mode='native-list',target=point['target']))
            del algo,searcher
        return
    rows=[]
    for point in job['points']:
        searcher=CoreSearcher(job['index'],index_params,point['parameters'])
        # One complete warmup. Native-loop timing removes Python dispatch,
        # retaining the same native search and result allocation as the bridge.
        ids=np.asarray([searcher.search_list(q,10) for q in queries],dtype=np.int64)
        assert ids.shape==truth.shape
        timings=[searcher.time_native(queries,10) for _ in range(7)]
        row=dict(label=label,profile=profile,dataset=job['dataset'],cycle=cycle,
                 target=point['target'],parameters=point['parameters'],index=job['index'],
                 query_count=len(queries),native_seconds=timings,
                 native_qps=len(queries)/float(np.median(timings)),
                 strict_id_recall=sum(len(set(a)&set(b)) for a,b in zip(ids,truth))/ids.size,
                 ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),module=zvec.__file__)
        rows.append(row)
        np.save(OUT/'measurements'/f'{profile}-{label}-{cycle}-{point["target"]}.npy',ids)
        print(json.dumps(row),flush=True)
        del searcher
    save(OUT/'measurements'/f'{profile}-{label}-{cycle}.json',rows)


def main(hold=False):
    state=json.loads((OUT/'paused-controller.json').read_text())
    controller=psutil.Process(state['pid'])
    assert abs(controller.create_time()-state['created'])<.01
    assert controller.status()==psutil.STATUS_STOPPED
    try:
        while any(p.status()!=psutil.STATUS_ZOMBIE for p in controller.children(recursive=True)):
            save(OUT/'progress.json',dict(phase='waiting_for_current_query',time=time.time()))
            time.sleep(5)
        plan=[]
        for profile in ['sift-exploration-v2','gist-exploration-v1','glove-exploration-v2']:
            job=json.loads((CAMPAIGN/'jobs'/(profile+'.json')).read_text())
            file=CAMPAIGN/'measurements/sweep'/profile/'portable/0/native-list/measurements.jsonl'
            rows=[json.loads(line) for line in file.read_text().splitlines()]
            assert len(rows)==len(job['queries'])
            points=[]
            for target in [.9,.95,.99]:
                best=max((r for r in rows if r['recall']>=target),key=lambda r:r['qps'])
                points.append(dict(target=target,parameters=best['parameters']))
            plan.append(dict(profile=profile,dataset=job['dataset'],index=job['directory']+'/index',points=points))
        save(OUT/'plan.json',plan)
        for label in ['control','fixed','prefetch','both']:
            save(OUT/'progress.json',dict(phase='building_ablation',label=label,time=time.time()))
            test_log=OUT/'builds'/label/'tests.log'
            if not test_log.exists() or '24 passed' not in test_log.read_text():
                rebuild(label)
        (OUT/'measurements').mkdir(exist_ok=True)
        labels=['exploration','succinct','control','fixed','prefetch','both']
        for cycle,order in enumerate([labels,list(reversed(labels))]):
            for job in plan:
                for label in order:
                    save(OUT/'progress.json',dict(phase='measuring',label=label,profile=job['profile'],cycle=cycle,time=time.time()))
                    with (OUT/f'{job["profile"]}-{label}-{cycle}.log').open('w') as log:
                        subprocess.run([sys.executable,__file__,'--measure','--label',label,
                            '--profile',job['profile'],'--cycle',str(cycle)],env=environment(label),
                            stdout=log,stderr=subprocess.STDOUT,check=True)
        if hold:
            save(OUT/'progress.json',dict(phase='awaiting_analysis',time=time.time()))
            deadline=time.monotonic()+1800
            while not (OUT/'release').exists():
                if time.monotonic()>deadline:
                    raise TimeoutError('Analysis hold exceeded 30 minutes; restoring campaign')
                time.sleep(2)
        save(OUT/'progress.json',dict(phase='finished',time=time.time()))
    except BaseException as exc:
        save(OUT/'progress.json',dict(phase='failed',error=repr(exc),time=time.time()))
        raise
    finally:
        if abs(controller.create_time()-state['created'])<.01:
            controller.send_signal(signal.SIGCONT)
            save(OUT/'controller-resumed.json',dict(pid=controller.pid,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--measure',action='store_true')
    p.add_argument('--label')
    p.add_argument('--profile')
    p.add_argument('--cycle',type=int,default=0)
    p.add_argument('--hold',action='store_true')
    p.add_argument('--official',action='store_true')
    args=p.parse_args()
    if args.measure:
        measure(args.label,args.profile,args.cycle,args.official)
    else:
        main(args.hold)
