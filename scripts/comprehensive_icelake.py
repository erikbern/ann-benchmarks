"""Isolated full-dataset SymQG sweep, paired optimization tests and baselines."""
import concurrent.futures
import csv
from datetime import datetime, timezone
import json
import os
import queue
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import docker
import psutil
import pybind11

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ann_benchmarks.definitions import Definition
from ann_benchmarks.results import build_result_filepath

CAMPAIGN = ROOT / 'results/comprehensive-icelake-20261116'
WORKER = ROOT / 'scripts/comprehensive_icelake_worker.py'
SOURCES = {v: Path.home() / 'workspace/zvec' / name / 'zvec' for v, name in (
    ('exploration', 'quantized_graph_exploration'), ('succinct', 'quantized_graph_exploration_succinct'))}
paused = []
children = []
owns_campaign = False


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def phase(name, **extra):
    value = dict(phase=name, time=datetime.now(timezone.utc).isoformat(), **extra)
    write(CAMPAIGN / 'progress.json', value)
    print(json.dumps(value), flush=True)


def environment(variant=None, build='portable'):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', ZVEC_TEST_SYMQG='1')
    env['PATH'] = ':'.join([str(ROOT / '.venv/bin'), str(ROOT / 'vendor/huawei/bin'), env['PATH']])
    env['PYTHONPATH'] = ':'.join([str(ROOT), str(SOURCES['exploration'] / 'py4bench/src')])
    if variant:
        env['PYTHONPATH'] = str(CAMPAIGN / 'overlays' / f'{variant}-{build}') + ':' + env['PYTHONPATH']
    env['LD_LIBRARY_PATH'] = ':'.join(str(ROOT / p) for p in [
        'vendor/runtime/usr/lib/x86_64-linux-gnu', 'vendor/runtime/usr/lib/x86_64-linux-gnu/blas',
        'vendor/runtime/usr/lib/x86_64-linux-gnu/lapack', 'vendor/huawei/lib', 'vendor/huawei/lib/x86_64-linux-gnu'])
    env['QSG_CREATE_SCRIPT'] = str(ROOT / 'vendor/huawei/create.py')
    return env


def start(command, name, env=None, cwd=ROOT):
    log = (CAMPAIGN / 'logs' / (name + '.log')).open('w')
    p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env or environment(),
                         cwd=cwd, start_new_session=True)
    log.close()
    children.append(p)
    return p


def run(command, name, env=None, cwd=ROOT):
    p = start(command, name, env, cwd)
    code = p.wait()
    if code:
        raise RuntimeError(f'{name} failed ({code}); see logs/{name}.log')


def freeze_dispatchers(recover=()):
    for p in psutil.process_iter(['cmdline', 'name', 'status']):
        cmd = p.info['cmdline'] or []
        if not p.info['name'].startswith('python') or not cmd or str(ROOT / '.venv') not in cmd[0]:
            continue
        if any(x.endswith('run_all_algorithms.py') for x in cmd) or ('run.py' in cmd and '--dataset' in cmd):
            already_owned = any(entry['pid'] == p.pid and
                                abs(entry['created'] - p.create_time()) < .01
                                for entry in recover)
            if p.info['status'] == psutil.STATUS_STOPPED and not already_owned:
                raise RuntimeError(f'Dispatcher {p.pid} already stopped by another task')
            paused.append(dict(pid=p.pid, created=p.create_time()))
            write(CAMPAIGN / 'paused-dispatchers.json', paused)
            p.send_signal(signal.SIGSTOP)


def plan():
    profiles = []
    old_profiles = {}
    for path in (ROOT / 'results/succinct-20261116/jobs').glob('*.json'):
        j = json.loads(path.read_text())
        old_profiles[j['id'].replace('succinct', 'zvec')] = j['definition']['query_argument_groups']
    for variant, source in SOURCES.items():
        imported = json.loads((ROOT / f'plan_{variant}.json').read_text())
        for j in imported['jobs']:
            _, dim, _, codec, _ = j['definition']['arguments']
            groups = list(old_profiles[j['profile']])
            groups.extend(q for q in j['definition']['query_argument_groups'] if q not in groups)
            identifier = j['profile'].replace('zvec', variant)
            directory = CAMPAIGN / 'indexes' / identifier
            directory.mkdir(parents=True)
            job = dict(id=identifier, dataset=j['dataset'], views=codec['rotation_views'], source=str(source),
                       variant=variant, directory=str(directory), campaign=str(CAMPAIGN), query_cpu=48,
                       cpus=list(range(2 + len(profiles) % 4 * 16, min(16 + len(profiles) % 4 * 16, 64), 2)),
                       queries=[q[0] for q in groups], vectors=str(CAMPAIGN / 'vectors' / (j['dataset'] + '.vecs')))
            if dim == 128:
                prior = 'three-datasets-20261113' if variant == 'exploration' else 'succinct-20261116'
                raw = list((ROOT / 'results' / prior / '.zvec-base-cache').glob('*/raw.index'))
                assert len(raw) == 1
                job['raw_index'] = str(raw[0])
            profiles.append(job)
    baseline_rows = list(csv.DictReader((ROOT / 'results/three-datasets-20261113/summary.csv').open()))
    selected = set()
    for dataset in {j['dataset'] for j in profiles}:
        for alg in ['qsgngt', 'descartes(01AI)']:
            for metric in ['recall', 'strict_id_recall']:
                for threshold in [.9, .95, .99]:
                    eligible = [r for r in baseline_rows if r['dataset'] == dataset and r['algorithm'] == alg
                                and float(r[metric]) >= threshold]
                    if eligible:
                        r = max(eligible, key=lambda r:float(r['qps']))
                        selected.add((dataset, alg, Path(r['result']).name))
    baselines = []
    for path in (ROOT / 'results/three-datasets-20261113/jobs').glob('*.json'):
        j = json.loads(path.read_text())
        d = Definition(**j['definition'])
        if d.algorithm not in ['qsgngt', 'descartes(01AI)']:
            continue
        queries = [q for q in d.query_argument_groups if
                   (j['dataset'], d.algorithm, Path(build_result_filepath(j['dataset'], 10, d, q)).name) in selected]
        if not queries:
            continue
        j['definition']['query_argument_groups'] = queries
        j.update(directory=str(CAMPAIGN / 'baseline-status' / j['id']), campaign=str(CAMPAIGN),
                 work=str(CAMPAIGN / 'baseline-work' / j['id']), query_cpu=48,
                 cpus=[18 + 2 * len(baselines)])
        Path(j['directory']).mkdir(parents=True)
        Path(j['work']).mkdir(parents=True)
        baselines.append(j)
    for job in profiles + baselines:
        write(CAMPAIGN / 'jobs' / (job['id'] + '.json'), job)
    write(CAMPAIGN / 'plan.json', dict(profiles=profiles, baselines=baselines,
         sweep_points=sum(len(j['queries']) for j in profiles), build_variants=['portable', 'icelake'],
         paired_order=['portable', 'icelake', 'icelake', 'portable'], query_cpu=48, runs=5,
         selection_metrics=['recall', 'strict_id_recall'], thresholds=[.9, .95, .99],
         paired_modes=['legacy', 'native-list'], note='Full query sets. Baselines rerun prior winners for both recall metrics.'))
    return profiles, baselines


def prepare_native():
    source = SOURCES['succinct']
    build_env = environment()
    # CMake ProcessorCount observes OMP_NUM_THREADS on this platform; do not
    # accidentally serialize nested dependency builds with the query setting.
    build_env['OMP_NUM_THREADS'] = '12'
    run([str(ROOT / '.venv/bin/cmake'), '-S', str(source), '-B', str(source / 'build-icelake'), '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', '-DENABLE_ICELAKE=ON', '-DBUILD_PY4BENCH=ON',
         '-DBUILD_PYTHON_BINDINGS=ON', '-DBUILD_TOOLS=ON', '-DBUILD_TESTING=OFF',
         '-DPython_EXECUTABLE=' + sys.executable, '-Dpybind11_DIR=' + pybind11.get_cmake_dir()], 'succinct-configure', build_env)
    run(['taskset', '-c', '32,34,36,38,40,42,44,46,50,52,54,56', str(ROOT / '.venv/bin/cmake'),
         '--build', str(source / 'build-icelake'), '--target', '_zvec', '-j', '12'], 'succinct-compile', build_env)
    for variant, source in SOURCES.items():
        for label, build in [('portable', 'build-ann'), ('icelake', 'build-icelake')]:
            overlay = CAMPAIGN / 'overlays' / f'{variant}-{label}' / 'zvec'
            shutil.copytree(source / 'python/zvec', overlay, dirs_exist_ok=True)
            binaries = list((source / build / 'lib').glob('_zvec*.so'))
            assert len(binaries) == 1
            shutil.copy2(binaries[0], overlay)
            run([sys.executable, '-m', 'pytest', '-o', 'addopts=', str(SOURCES['exploration'] / 'py4bench/tests'), '-q'],
                f'tests-{variant}-{label}', environment(variant, label))


def prepare_indexes(profiles):
    converter = SOURCES['exploration'] / 'build-ann/bin/txt2vecs'
    for dataset in sorted({j['dataset'] for j in profiles if not j['dataset'].startswith('sift')}):
        run([sys.executable, str(WORKER), '--action', 'convert', '--dataset', str(ROOT / 'data' / (dataset + '.hdf5')),
             '--output', str(CAMPAIGN / 'vectors' / (dataset + '.vecs')), '--converter', str(converter)], 'convert-' + dataset)
    slots = queue.Queue()
    for slot in range(4):
        slots.put(list(range(2 + slot * 16, min(16 + slot * 16, 64), 2)))

    def build_one(job):
        cpus = slots.get()
        try:
            job['cpus'] = cpus
            path = CAMPAIGN / 'jobs' / (job['id'] + '.json')
            write(path, job)
            run([sys.executable, str(WORKER), '--action', 'build', '--job', str(path)], 'build-' + job['id'])
        finally:
            slots.put(cpus)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(build_one, j) for j in sorted(profiles, key=lambda j:j['dataset'].startswith('sift'))]
        for f in concurrent.futures.as_completed(pending):
            f.result()


def prepare_baselines(jobs):
    workers = []
    for j in jobs:
        prior = ROOT / 'results/three-datasets-20261113/work' / j['id']
        # Clone persistent vendor caches into this campaign before invoking fit;
        # original baseline artifacts remain untouched even if a vendor writes.
        for directory in prior.iterdir():
            if directory.is_dir() and not directory.is_symlink():
                if not (Path(j['work']) / directory.name).exists():
                    run(['cp', '-a', '--reflink=auto', str(directory), j['work']], 'copy-' + j['id'] + '-' + directory.name)
        p = start([sys.executable, str(WORKER), '--action', 'baseline', '--job',
                   str(CAMPAIGN / 'jobs' / (j['id'] + '.json'))], 'baseline-' + j['id'])
        workers.append((j, p))
    while any(not (Path(j['directory']) / 'ready.json').exists() for j, _ in workers):
        for j, p in workers:
            if p.poll() is not None:
                raise RuntimeError('Baseline builder failed: ' + j['id'])
        time.sleep(5)
    return workers


def drain():
    client = docker.from_env()
    timed_out = []
    while True:
        active = [c for c in client.containers.list() if '--dataset' in (c.attrs['Config'].get('Cmd') or [])]
        if not active:
            break
        phase('draining_prior_jobs', containers=[c.short_id for c in active])
        for c in active:
            started = datetime.strptime(c.attrs['State']['StartedAt'][:19], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
            if time.time() - started >= 7200:
                try:
                    c.kill()
                    timed_out.append(c.id)
                    write(CAMPAIGN / 'prior_job_timeouts.json', timed_out)
                except docker.errors.NotFound:
                    pass
        time.sleep(10)
    write(CAMPAIGN / 'isolation-start.json', dict(time=datetime.now(timezone.utc).isoformat(), active_benchmark_containers=0))


def main():
    global owns_campaign
    if CAMPAIGN.exists():
        raise RuntimeError('Campaign already exists; refusing to overwrite')
    for name in ['logs', 'jobs', 'vectors', 'indexes', 'overlays']:
        (CAMPAIGN / name).mkdir(parents=True, exist_ok=True)
    owns_campaign = True
    write(CAMPAIGN / 'controller.json', dict(pid=os.getpid(), created=psutil.Process().create_time()))
    freeze_dispatchers()
    profiles, baselines = plan()
    phase('preparing', profiles=len(profiles), sweep_points=sum(len(j['queries']) for j in profiles), baselines=len(baselines))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        native = pool.submit(prepare_native)
        indexes = pool.submit(prepare_indexes, profiles)
        baseline = pool.submit(prepare_baselines, baselines)
        native.result()
        indexes.result()
        workers = baseline.result()
    phase('waiting_for_isolation')
    drain()
    for j, p in workers:
        phase('measuring_baseline', job=j['id'])
        (Path(j['directory']) / 'query').touch()
        if p.wait():
            raise RuntimeError('Baseline query failed: ' + j['id'])
    for j in profiles:
        jobfile = str(CAMPAIGN / 'jobs' / (j['id'] + '.json'))
        phase('sweeping', job=j['id'], points=len(j['queries']))
        run([sys.executable, str(WORKER), '--action', 'query', '--job', jobfile],
            'sweep-' + j['id'], environment(j['variant']))
        for cycle, build in enumerate(['portable', 'icelake', 'icelake', 'portable']):
            phase('paired_validation', job=j['id'], build=build, cycle=cycle)
            run([sys.executable, str(WORKER), '--action', 'query', '--job', jobfile,
                 '--native-build', build, '--phase', 'paired', '--cycle', str(cycle)],
                f"paired-{j['id']}-{cycle}-{build}", environment(j['variant'], build))
    phase('summarizing')
    run([sys.executable, str(ROOT / 'scripts/summarize_comprehensive_icelake.py'), str(CAMPAIGN)], 'summary')
    phase('finished')


def stop(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except BaseException as exc:
        if owns_campaign:
            phase('failed', error=repr(exc))
        for p in children:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        raise
    finally:
        for entry in paused:
            try:
                p = psutil.Process(entry['pid'])
                if abs(p.create_time() - entry['created']) < .01:
                    p.send_signal(signal.SIGCONT)
            except psutil.NoSuchProcess:
                pass
        if owns_campaign:
            write(CAMPAIGN / 'dispatchers-resumed.json', dict(time=datetime.now(timezone.utc).isoformat(), processes=paused))
