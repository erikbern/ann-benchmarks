"""Sample worker process-tree RSS without treating it as net index size."""
import json
from pathlib import Path
import sys
import time

import psutil

campaign = Path(sys.argv[1]).resolve()
workers = {}
while True:
    for p in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        cmd = p.info['cmdline'] or []
        if not any(x.endswith('comprehensive_icelake_worker.py') for x in cmd) or '--job' not in cmd:
            continue
        job = Path(cmd[cmd.index('--job') + 1])
        if job.parent != campaign / 'jobs':
            continue
        key = str(p.pid) + '-' + str(p.info['create_time'])
        row = workers.setdefault(key, dict(pid=p.pid, job=job.stem, command=cmd,
                                         first_observed=time.time(), peak_rss_bytes=0, peak_tree_rss_bytes=0))
        try:
            rss = p.memory_info().rss
            tree = rss
            for child in p.children(recursive=True):
                try:
                    tree += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            row.update(last_observed=time.time(), peak_rss_bytes=max(row['peak_rss_bytes'], rss),
                       peak_tree_rss_bytes=max(row['peak_tree_rss_bytes'], tree))
        except psutil.NoSuchProcess:
            pass
    tmp = campaign / 'telemetry.tmp'
    tmp.write_text(json.dumps(dict(note='5-second sampled worker RSS including input arrays, index, and query results; not net index memory. Monitoring started after campaign launch.', workers=list(workers.values())), indent=2))
    tmp.replace(campaign / 'telemetry.json')
    phase = json.loads((campaign / 'progress.json').read_text())['phase']
    if phase in ['finished', 'failed']:
        break
    time.sleep(5)
