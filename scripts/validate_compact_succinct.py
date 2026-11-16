"""Build/test a compact candidate, then confirm it against a fresh control."""
import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import sys

from diagnose_succinct import OUT, ROOT, SUCCINCT, BUILD, rebuild, environment, save
from prepare_compact_succinct import prepare


def kernel_tests():
    gtest=SUCCINCT/'thirdparty/googletest/googletest-1.10.0/googletest'
    includes=[gtest/'include',gtest,SUCCINCT/'src',SUCCINCT/'src/include',
              SUCCINCT/'thirdparty/RaBitQ-Library/RaBitQ-Library-0.7.0/include']
    test_source=SUCCINCT/'tests/turbo/turbo_symqg_kernels_test.cc'
    source=[gtest/'src/gtest-all.cc',gtest/'src/gtest_main.cc',test_source]
    if 'ScanSpecializationsMatchIndependentAccumulator' not in test_source.read_text():
        source.append(OUT/'compact-source/scan_reference_test.cc')
    for label in ['control','compact']:
        turbo=OUT/'builds'/label/'libzvec_turbo.a'
        if not turbo.exists():
            turbo=BUILD/'lib/libzvec_turbo.a'
        binary=OUT/'builds'/label/'scan-tests'
        cmd=['c++','-O2','-std=c++17','-ffp-contract=off','-DRABITQ_SUPPORTED=1',
             *['-I'+str(p) for p in includes],*[str(p) for p in source],
             '-Wl,--start-group',str(turbo),str(BUILD/'lib/libzvec_ailego.a'),
             str(BUILD/'lib/librabitq_fastscan.a'),str(BUILD/'lib/librabitq_core.a'),
             '-Wl,--end-group','-pthread','-ldl','-o',str(binary)]
        save(OUT/'builds'/label/'scan-test-command.json',cmd)
        with (OUT/'builds'/label/'scan-tests-build.log').open('w') as log:
            subprocess.run(['taskset','-c','50',*cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
        with (OUT/'builds'/label/'scan-tests.log').open('w') as log:
            subprocess.run(['taskset','-c','48',str(binary)],stdout=log,stderr=subprocess.STDOUT,check=True)


def main(scan,prefetch,official=False):
    assert scan or prefetch
    save(OUT/'compact-progress.json',dict(phase='building',scan=scan,prefetch=prefetch))
    rebuild('compact',kernel=prepare() if scan else None,prefetch_override=prefetch)
    kernel_tests()
    alias=OUT/'builds/control-confirm'
    alias.mkdir(exist_ok=True)
    if not (alias/'overlay').exists():
        (alias/'overlay').symlink_to(OUT/'builds/control/overlay',target_is_directory=True)
    assert (alias/'overlay').resolve()==OUT/'builds/control/overlay'
    fingerprint=json.loads((OUT/'builds/compact/binary.json').read_text())
    manifest=OUT/'compact-validation-binary.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==fingerprint, 'Use a fresh artifact directory when changing the candidate'
    save(manifest,fingerprint)
    plan=json.loads((OUT/'plan.json').read_text())
    for cycle,labels in enumerate([['control-confirm','compact'],['compact','control-confirm']]):
        for job in plan:
            for label in labels:
                save(OUT/'compact-progress.json',dict(phase='measuring',profile=job['profile'],label=label,cycle=cycle))
                with (OUT/f'confirm-{job["profile"]}-{label}-{cycle}.log').open('w') as log:
                    subprocess.run([sys.executable,str(ROOT/'scripts/diagnose_succinct.py'),
                        '--measure','--label',label,'--profile',job['profile'],'--cycle',str(cycle)],
                        env=environment(label),stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=[]
    for job in plan:
        rows={label:sum((json.loads((OUT/'measurements'/f'{job["profile"]}-{label}-{cycle}.json').read_text())
                        for cycle in [0,1]),[]) for label in ['compact','control-confirm']}
        for target in [.9,.95,.99]:
            a=[r for r in rows['control-confirm'] if r['target']==target]
            b=[r for r in rows['compact'] if r['target']==target]
            assert len(a)==len(b)==2
            assert len({r['ids_sha256'] for r in a+b})==1,('Result changed',job['profile'],target)
            old=median(r['native_qps'] for r in a)
            new=median(r['native_qps'] for r in b)
            summary.append(dict(dataset=job['dataset'],target=target,parameters=a[0]['parameters'],
                control_native_qps=old,compact_native_qps=new,gain_pct=100*(new/old-1),
                control_repeat_qps=[r['native_qps'] for r in a],compact_repeat_qps=[r['native_qps'] for r in b],
                strict_id_recall=a[0]['strict_id_recall'],ids_identical=True))
    save(OUT/'compact-summary.json',summary)
    save(OUT/'compact-progress.json',dict(phase='native_finished' if official else 'finished',scan=scan,prefetch=prefetch))
    for r in summary:
        print(r['dataset'],r['target'],round(r['gain_pct'],2),'%',flush=True)
    if official:
        for cycle,labels in enumerate([['control-confirm','compact'],['compact','control-confirm']]):
            for job in plan:
                for label in labels:
                    save(OUT/'compact-progress.json',dict(phase='official_measurement',profile=job['profile'],label=label,cycle=cycle))
                    with (OUT/f'official-{job["profile"]}-{label}-{cycle}.log').open('w') as log:
                        subprocess.run([sys.executable,str(ROOT/'scripts/diagnose_succinct.py'),
                            '--measure','--official','--label',label,'--profile',job['profile'],'--cycle',str(cycle)],
                            env=environment(label),stdout=log,stderr=subprocess.STDOUT,check=True)
        records=[]
        for file in (OUT/'official').rglob('measurements.jsonl'):
            records.extend(json.loads(line) for line in file.read_text().splitlines())
        summary=[]
        for job in plan:
            for target in [.9,.95,.99]:
                points=[r for r in records if r['profile']==job['profile'] and r['target']==target]
                assert len(points)==4
                assert len({r['ids_sha256'] for r in points})==1
                assert min(r['recall'] for r in points)>=target
                a=[r for r in points if r['variant']=='control-confirm']
                b=[r for r in points if r['variant']=='compact']
                old,new=median(r['qps'] for r in a),median(r['qps'] for r in b)
                summary.append(dict(dataset=job['dataset'],target=target,
                    control_qps=old,compact_qps=new,gain_pct=100*(new/old-1),
                    control_repeat_qps=[r['qps'] for r in a],compact_repeat_qps=[r['qps'] for r in b],
                    recall=points[0]['recall'],strict_id_recall=points[0]['strict_id_recall'],
                    ids_identical=True,parameters=points[0]['parameters']))
        save(OUT/'compact-official-summary.json',summary)
        save(OUT/'compact-progress.json',dict(phase='finished',scan=scan,prefetch=prefetch,official=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--scan',action='store_true')
    p.add_argument('--prefetch',action='store_true')
    p.add_argument('--official',action='store_true')
    a=p.parse_args()
    main(a.scan,a.prefetch,a.official)
