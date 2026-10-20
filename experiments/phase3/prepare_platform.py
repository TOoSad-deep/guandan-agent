"""Reproduce the isolated platform build; original research snapshots stay intact."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SNAPSHOT = PROJECT / 'research' / 'sources' / 'OpenGuanDan-Simulator-e92db7f'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--java-home', type=Path, help='Use JDK 17 or 21; this snapshot uses Lombok 1.18.30.')
    args = parser.parse_args()
    results = ROOT / 'results'
    results.mkdir(parents=True, exist_ok=True)
    for name in ['guandan-websocket', 'guandan']:
        dest = ROOT / 'workspace' / name
        if not dest.exists():
            shutil.copytree(SNAPSHOT / name, dest,
                            ignore=shutil.ignore_patterns('target', '.git', '__pycache__', 'log.txt'))
    server_file = ROOT / 'workspace/guandan-websocket/src/main/java/com/guandan/socket/NettyWebSocketServer.java'
    before = 'bootstrap.bind(config.getServerPort())'
    after = 'bootstrap.bind("127.0.0.1", config.getServerPort())'
    source = server_file.read_text()
    if before not in source and after not in source:
        raise RuntimeError('Unexpected server source; inspect before applying local bind adaptation.')
    server_file.write_text(source.replace(before, after))
    env = os.environ.copy()
    if args.java_home:
        env['JAVA_HOME'] = str(args.java_home.resolve())
    commands = []
    for name, log_name in [('guandan-websocket', 'rebuild-server.log'), ('guandan', 'rebuild-action-engine.log')]:
        command = ['mvn', '-B', '-ntp', '-f', str(ROOT / 'workspace' / name / 'pom.xml'),
                   '-Dmaven.repo.local=' + str(ROOT / '.cache/m2'), 'package']
        with (results / log_name).open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        commands.append({'command': command, 'exit_code': result.returncode, 'log': log_name})
        if result.returncode:
            raise SystemExit(f'{name} failed; inspect {results / log_name}')
    (results / 'rebuild-commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    print('Built both isolated artifacts. Only server loopback binding was adapted.')


if __name__ == '__main__':
    main()
