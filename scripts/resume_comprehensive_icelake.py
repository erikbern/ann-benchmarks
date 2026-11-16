"""Resume a failed suite from immutable snapshots and completed query records."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import psutil
import comprehensive_icelake as c
owns_campaign = False


def complete(job, phase, build, cycle, params):
    root = c.CAMPAIGN / 'measurements' / phase / job['id'] / build / str(cycle)
    modes = ['native-list'] if phase == 'sweep' else ['legacy', 'native-list']
    for mode in modes:
        file = root / mode / 'measurements.jsonl'
        if not file.exists():
            return False
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        if len(rows) != len(params) or any(not Path(r['result']).exists() for r in rows):
            return False
        if any(not any(r['parameters'] == q for r in rows) for q in params):
            return False
    return phase != 'sweep' or (Path(job['directory']) / 'selected.json').exists()


def main():
    global owns_campaign
    state = json.loads((c.CAMPAIGN / 'controller.json').read_text())
    try:
        p = psutil.Process(state['pid'])
        if abs(p.create_time()-state['created']) < .01 and p.status() != psutil.STATUS_ZOMBIE:
            raise RuntimeError('Original controller still running; refusing a concurrent suite')
    except psutil.NoSuchProcess:
        pass
    if json.loads((c.CAMPAIGN / 'progress.json').read_text())['phase'] == 'finished':
        print('Suite already finished')
        return
    plan = json.loads((c.CAMPAIGN / 'plan.json').read_text())
    for j in plan['profiles']:
        assert (Path(j['directory']) / 'ready.json').exists(), j['id']
    versions = json.loads((c.CAMPAIGN / 'binaries.json').read_text())
    for variant in versions.values():
        for binary in variant['builds'].values():
            assert hashlib.sha256(Path(binary['binary']).read_bytes()).hexdigest() == binary['sha256']
    for snapshot in json.loads((c.CAMPAIGN / 'index-manifest.json').read_text()):
        digest = hashlib.sha256()
        with (Path(snapshot['job']['directory']) / 'index').open('rb') as f:
            while block := f.read(8*1024*1024):
                digest.update(block)
        assert digest.hexdigest() == snapshot['sha256'], snapshot['profile']
    c.write(c.CAMPAIGN / 'controller.json', dict(pid=os.getpid(), created=psutil.Process().create_time()))
    owns_campaign = True
    # The dead controller may have been killed before its finally block ran.
    # Adopt only the exact PID/create-time pairs that campaign had suspended.
    previous = c.CAMPAIGN / 'paused-dispatchers.json'
    c.freeze_dispatchers(json.loads(previous.read_text()) if previous.exists() else ())
    jobs = [j for j in plan['baselines'] if not (Path(j['directory']) / 'done').exists()]
    for j in jobs:
        for name in ['query','ready.json']:
            (Path(j['directory']) / name).unlink(missing_ok=True)
    workers = c.prepare_baselines(jobs)
    c.drain()
    for j,p in workers:
        c.phase('measuring_baseline', job=j['id'])
        (Path(j['directory']) / 'query').touch()
        if p.wait():
            raise RuntimeError('Baseline failed: ' + j['id'])
    for j in plan['profiles']:
        jobfile = str(c.CAMPAIGN / 'jobs' / (j['id']+'.json'))
        if not complete(j,'sweep','portable',0,j['queries']):
            c.phase('sweeping',job=j['id'])
            c.run([sys.executable,str(c.WORKER),'--action','query','--job',jobfile],
                  'resume-sweep-'+j['id'],c.environment(j['variant']))
        selected = json.loads((Path(j['directory']) / 'selected.json').read_text())
        for cycle,build in enumerate(['portable','icelake','icelake','portable']):
            if complete(j,'paired',build,cycle,selected):
                continue
            c.phase('paired_validation',job=j['id'],build=build,cycle=cycle)
            c.run([sys.executable,str(c.WORKER),'--action','query','--job',jobfile,
                   '--native-build',build,'--phase','paired','--cycle',str(cycle)],
                  f"resume-paired-{j['id']}-{cycle}",c.environment(j['variant'],build))
    c.run([sys.executable,str(c.ROOT/'scripts/summarize_comprehensive_icelake.py'),str(c.CAMPAIGN)],'summary')
    c.phase('finished')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM,c.stop)
    signal.signal(signal.SIGINT,c.stop)
    try:
        main()
    except BaseException as exc:
        if owns_campaign:
            c.phase('failed',error=repr(exc))
        for p in c.children:
            if p.poll() is None:
                try:
                    os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
        raise
    finally:
        for entry in c.paused:
            try:
                p=psutil.Process(entry['pid'])
                if abs(p.create_time()-entry['created'])<.01:
                    p.send_signal(signal.SIGCONT)
            except psutil.NoSuchProcess:
                pass
        if owns_campaign:
            c.write(c.CAMPAIGN / 'dispatchers-resumed.json', dict(
                time=c.datetime.now(c.timezone.utc).isoformat(), processes=c.paused))
