"""Diagnostic-only header instrumentation, built outside the source tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from diagnose_glove_gap import OUT, PRIOR, ROOT, save
from diagnose_succinct import SUCCINCT, BUILD, environment


def build():
    # Called by the isolated coordinator after the broad margin grid. Resolve
    # its narrow threshold brackets before profiling the final selected points.
    subprocess.run([sys.executable,str(ROOT/'scripts/refine_glove_margin.py')],check=True)
    folder=OUT/'profile-build'
    include=folder/'include'
    header='algorithm/quantized_graph/quantized_graph.h'
    target=include/header
    target.parent.mkdir(parents=True,exist_ok=True)
    original=(SUCCINCT/'src/core'/header).read_text()
    changed=original.replace('#include <cmath>','#include <cmath>\n#include <chrono>\n#include <cstdio>')
    changed=changed.replace('  auto &results = ctx.reset_results();',
        '  using ProfileClock = std::chrono::steady_clock;\n'
        '  const auto p0 = ProfileClock::now();\n'
        '  size_t profile_exact = 0;\n'
        '  auto &results = ctx.reset_results();',1)
    changed=changed.replace('  const auto &options = ctx.quantized_graph_options();',
        '  const auto p1 = ProfileClock::now();\n'
        '  const auto &options = ctx.quantized_graph_options();',1)
    changed=changed.replace('  beam.reset();','  const auto p2 = ProfileClock::now();\n  beam.reset();',1)
    changed=changed.replace('  // Exact Top-K of the expanded set,',
        '  const auto p3 = ProfileClock::now();\n  // Exact Top-K of the expanded set,',1)
    changed=changed.replace('      const float exact = ctx.distance(raw[i].data());',
        '      ++profile_exact;\n      const float exact = ctx.distance(raw[i].data());',1)
    changed=changed.replace("  // As in SymphonyQG's result completion step,",
        "  const auto p4 = ProfileClock::now();\n  // As in SymphonyQG's result completion step,",1)
    ending='  return 0;\n}\n\n}  // namespace zvec::core'
    instrumentation=r'''  const auto p5 = ProfileClock::now();
  struct ProfileTotals { size_t n=0, expanded=0, candidates=0, exact=0; double us[5]{}; };
  static thread_local ProfileTotals totals;
  const ProfileClock::time_point stamps[]{p0,p1,p2,p3,p4,p5};
  for (size_t i=0;i<5;++i)
    totals.us[i] += std::chrono::duration<double,std::micro>(stamps[i+1]-stamps[i]).count();
  totals.expanded += stopping.expanded;
  totals.candidates += refine.size();
  totals.exact += profile_exact;
  if (++totals.n == 10000) {
    std::fprintf(stderr,"GLOVE_PROFILE %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
        totals.us[0]/totals.n,totals.us[1]/totals.n,totals.us[2]/totals.n,
        totals.us[3]/totals.n,totals.us[4]/totals.n,
        double(totals.expanded)/totals.n,double(totals.candidates)/totals.n,double(totals.exact)/totals.n);
    totals = {};
  }
'''
    assert changed.count(ending)==1
    changed=changed.replace(ending,instrumentation+ending)
    target.write_text(changed)
    commands=json.loads((PRIOR/'original-build-commands.json').read_text())
    replacements={}
    recorded=[]
    for obj_path,archive,label in [
        ('src/core/algorithm/hnsw/CMakeFiles/core_knn_hnsw_objects.dir/hnsw_quantized_graph.cc.o','lib/libcore_knn_hnsw.a','hnsw'),
        ('src/core/CMakeFiles/zvec_core.dir/algorithm/hnsw/hnsw_quantized_graph.cc.o','lib/libzvec_core.a','core')]:
        obj=folder/label/'hnsw_quantized_graph.cc.o'
        obj.parent.mkdir(exist_ok=True)
        cmd=shlex.split(commands[obj_path])
        cmd[1:1]=['-I'+str(include),'-I'+str(SUCCINCT/'src/core/algorithm/quantized_graph')]
        for flag,value in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:
            cmd[cmd.index(flag)+1]=value
        recorded.append(cmd)
        with (folder/(label+'.log')).open('w') as log:
            subprocess.run(['taskset','-c','50',*cmd],cwd=BUILD,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert str(target) in Path(str(obj)+'.d').read_text()
        copied=folder/Path(archive).name
        shutil.copy2(BUILD/archive,copied)
        subprocess.run(['ar','r',str(copied),str(obj)],check=True)
        replacements[archive]=str(copied)
    name='lib/_zvec.cpython-310-x86_64-linux-gnu.so'
    link=shlex.split(commands[name].split('&&')[1].strip())
    overlay=folder/'overlay/zvec'
    shutil.copytree(SUCCINCT/'python/zvec',overlay,dirs_exist_ok=True)
    binary=overlay/Path(name).name
    link[link.index('-o')+1]=str(binary)
    link=[replacements.get(arg,arg) for arg in link]
    recorded.append(link)
    with (folder/'link.log').open('w') as log:
        subprocess.run(['taskset','-c','50',*link],cwd=BUILD,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['strip',str(binary)],check=True)
    save('profile-build/commands.json',recorded)
    save('profile-build/binary.json',dict(binary=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    env=environment('compact')
    env['PYTHONPATH']=str(folder/'overlay')+':'+env['PYTHONPATH']
    with (OUT/'profile.log').open('w') as log:
        subprocess.run([sys.executable,__file__,'--measure'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def measure():
    import numpy as np
    import yaml
    from py4bench import CoreSearcher
    from comprehensive_icelake_worker import load_data
    from ann_benchmarks.distance import metrics
    os.sched_setaffinity(0,{48})
    train,queries,truth,thresholds,metric=load_data('glove-100-angular')
    method=json.loads((OUT/'method.json').read_text())
    confirmations=json.loads((OUT/'confirmations.json').read_text())
    points=list(method['selected_parameters'])
    for target in [.9,.95,.99]:
        eligible=[r for r in confirmations if r['target']==target and r['recall']>=target]
        best=max(eligible,key=lambda r:r['qps'])
        point=dict(target=target,parameters=best['parameters'])
        if point not in points: points.append(point)
    results=[]
    for point in points:
        print('PROFILE_POINT '+json.dumps(point),flush=True)
        searcher=CoreSearcher(method['index'],method['index_params'],point['parameters'])
        ids=np.asarray([searcher.search_list(q,10) for q in queries],dtype=np.int64)
        sha=hashlib.sha256(ids.tobytes()).hexdigest()
        expected=next(r['ids_sha256'] for r in confirmations if r['parameters']==point['parameters'])
        assert sha==expected, 'Instrumentation must not change search results'
        print('PROFILE_MODE native',flush=True)
        native=[searcher.time_native(queries,10) for _ in range(3)]
        print('PROFILE_MODE interleaved',flush=True)
        for _ in range(2):
            answers=[]
            for q in queries:
                row=searcher.search_list(q,10)
                answers.append([(int(i),float(metrics[metric].distance(q,train[i]))) for i in row])
        results.append(dict(**point,ids_identical=True,ids_sha256=sha,native_seconds=native))
        save('profile-results.json',results)
        del searcher
    # Run affinity control with the uninstrumented binary in another process;
    # profiling output must not influence the control's throughput.
    env=environment('compact')
    with (OUT/'affinity.log').open('w') as log:
        subprocess.run([sys.executable,__file__,'--affinity'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def affinity():
    from comprehensive_icelake_worker import load_data
    from refine_glove_margin import measure_affinity
    method=json.loads((OUT/'method.json').read_text())
    data=load_data('glove-100-angular')
    measure_affinity(method,data[1])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--measure',action='store_true')
    p.add_argument('--affinity',action='store_true')
    a=p.parse_args()
    if a.affinity: affinity()
    elif a.measure: measure()
    else: build()
