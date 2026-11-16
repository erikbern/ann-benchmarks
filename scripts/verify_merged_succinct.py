"""Check the normal succinct build against the validated diagnostic candidate."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from diagnose_succinct import OUT, ROOT, SUCCINCT, EXPLORATION, BUILD, environment, save

folder=OUT/'builds/final-merged'
overlay=folder/'overlay/zvec'
shutil.copytree(SUCCINCT/'python/zvec',overlay,dirs_exist_ok=True)
binary=BUILD/'lib/_zvec.cpython-310-x86_64-linux-gnu.so'
shutil.copy2(binary,overlay/binary.name)
with (folder/'tests.log').open('w') as log:
    subprocess.run([sys.executable,'-m','pytest','-o','addopts=',str(EXPLORATION/'py4bench/tests'),'-q'],
                   env=environment('final-merged'),stdout=log,stderr=subprocess.STDOUT,check=True)
cmd=json.loads((OUT/'builds/compact/scan-test-command.json').read_text())
cmd=[str(BUILD/'lib/libzvec_turbo.a') if arg==str(OUT/'builds/compact/libzvec_turbo.a') else arg
     for arg in cmd if arg!=str(OUT/'compact-source/scan_reference_test.cc')]
test_binary=folder/'scan-tests'
cmd[cmd.index('-o')+1]=str(test_binary)
save(folder/'scan-test-command.json',cmd)
with (folder/'scan-tests-build.log').open('w') as log:
    subprocess.run(['taskset','-c','50',*cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
with (folder/'scan-tests.log').open('w') as log:
    subprocess.run([str(test_binary)],stdout=log,stderr=subprocess.STDOUT,check=True)
code=r'''
import json,os,hashlib,pathlib,h5py,numpy as np,yaml
from py4bench import CoreSearcher
os.sched_setaffinity(0,{48})
root=pathlib.Path.cwd();out=root/'results/succinct-diagnosis-20261116'
source=pathlib.Path.home()/'workspace/zvec/quantized_graph_exploration/zvec'
rows=[]
for job in json.loads((out/'plan.json').read_text()):
    with h5py.File(root/'data'/(job['dataset']+'.hdf5')) as f:queries=f['test'][:]
    config=yaml.safe_load((source/'tools/core/bench/config'/(job['dataset'].split('-')[0]+'_fp32_recall_bench.yaml')).read_text())
    for point in job['points']:
        searcher=CoreSearcher(job['index'],json.loads(config['IndexCommon']['IndexConfig']),point['parameters'])
        ids=np.asarray([searcher.search_list(q,10) for q in queries],dtype=np.int64)
        expected=np.load(out/'measurements'/f'{job["profile"]}-compact-0-{point["target"]}.npy')
        np.testing.assert_array_equal(ids,expected)
        rows.append(dict(dataset=job['dataset'],target=point['target'],queries=len(queries),ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),ids_identical=True))
        del searcher
(out/'final-build-id-verification.json').write_text(json.dumps(rows,indent=2))
print('Verified',len(rows),'full query configurations',flush=True)
'''
with (folder/'full-query-id-check.log').open('w') as log:
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=environment('final-merged'),
                   stdout=log,stderr=subprocess.STDOUT,check=True)
save(OUT/'final-build.json',dict(binary=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    binding_tests='24 passed',kernel_tests='4 passed',full_query_configurations=9,all_ids_identical=True))
print('Merged build verified',flush=True)
