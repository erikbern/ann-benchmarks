#!/usr/bin/env python3
"""Build the Python QG bridge against this machine's validated native runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--variant', choices=['succinct', 'exploration'], default='succinct')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, output = a.source.resolve(), a.output.resolve()
    build = source / 'build-r7i'
    provenance = json.loads((root / 'benchmark-configs/r7i-new-api' / a.variant / 'validation/provenance.json').read_text())
    tested = provenance['zvec_measurement_commit']
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', tested, '--',
                    'src', 'cmake', 'CMakeLists.txt', 'tools/core/qg_bench.cc', 'tools/core/CMakeLists.txt'], check=True)
    if git('ls-files', '--others', '--exclude-standard', 'src'):
        raise SystemExit('Untracked core sources require a new core validation')
    if digest(build / 'bin/qg_bench') != provenance['binary']['sha256']:
        raise SystemExit('Native binary is not the validated runtime; rebuild and revalidate core first')
    entry = next(e for e in json.loads((build / 'compile_commands.json').read_text()) if e['file'].endswith('/qg_bench.cc'))
    args = shlex.split(entry['command'])
    output.mkdir(parents=True, exist_ok=False)
    cpp = root / 'ann_benchmarks/algorithms/zvec_qg/core.cpp'
    obj = output / 'core.o'
    extension = output / ('_zvec_qg_core' + sysconfig.get_config_var('EXT_SUFFIX'))
    args[args.index('-o') + 1] = str(obj)
    args[args.index('-c') + 1] = str(cpp)
    args = [arg for arg in args if arg != '-fPIE']
    args += ['-fPIC', '-fvisibility=hidden']
    args += shlex.split(subprocess.check_output([sys.executable, '-m', 'pybind11', '--includes'], text=True))
    ninja = shutil.which('ninja') or str(Path(sys.executable).parent / 'ninja')
    original = subprocess.check_output([ninja, '-C', str(build), '-t', 'commands', 'bin/qg_bench'], text=True).splitlines()[-1]
    link = shlex.split(original)
    if link[:2] != [':', '&&'] or link[-2:] != ['&&', ':']:
        raise SystemExit('Unrecognized native link command')
    link = link[2:-2]
    link[link.index('-o') + 1] = str(extension)
    link = [str(obj) if arg.endswith('/qg_bench.cc.o') else arg for arg in link if not arg.startswith('-Wl,-rpath,')]
    link += ['-shared', '-Wl,-rpath,' + str(build / 'lib')]
    libraries = {str((build / arg).resolve()) for arg in link if arg.endswith(('.a', '.so')) and arg != str(extension)}
    libraries.update(str(path.resolve()) for path in (build / 'lib').glob('*.so*') if path.is_file())
    before = {path: digest(path) for path in sorted(libraries)}
    commands = dict(compile=args, link=link, cwd=str(build))
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    for name, command in [('compile', args), ('link', link)]:
        with (output / (name + '.log')).open('w') as log:
            subprocess.run(command, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
    if any(digest(path) != sha for path, sha in before.items()):
        raise SystemExit('Runtime changed during build')
    artifact = dict(source_commit=git('rev-parse', 'HEAD'), validated_core_commit=tested,
                    source_tree=git('rev-parse', 'HEAD:src'), variant=a.variant,
                    extension=str(extension), python=sys.executable,
                    python_version=sys.version, bridge_sha256=digest(cpp),
                    native_binary_sha256=provenance['binary']['sha256'],
                    runtime_sha256={**before, str(extension): digest(extension)})
    (output / 'artifact.json').write_text(json.dumps(artifact, indent=2) + '\n')
    print(output / 'artifact.json')


if __name__ == '__main__':
    main()
