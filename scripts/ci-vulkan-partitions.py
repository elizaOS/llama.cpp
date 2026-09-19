#!/usr/bin/env python3
"""Account for complete Vulkan test invocations and time a bounded partition pilot.

Support mode supplies case identities, not a numerical support oracle. Every
selected occurrence needs an explicit terminal outcome, and previously witnessed
passing cases must still pass. A pilot never represents the unselected suite.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

SOURCE = 'ff3f6daa396b344f70f35019277c1238c6e09625'
LANES = {'broad': ['MUL_MAT', 'MUL_MAT_ID', 'ADD', 'MUL', 'SOFT_MAX', 'RMS_NORM', 'CPY', 'ROPE', 'FLASH_ATTN_EXT'],
         'custom': ['GET_ROWS', 'CPY', 'MUL_MAT', 'ATTN_SCORE_TBQ', 'ATTN_SCORE_POLAR', 'ISTFT']}
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def run(command, stem, seconds):
    start = time.monotonic()
    with Path(str(stem) + '.stdout').open('wb') as out, Path(str(stem) + '.stderr').open('wb') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    receipt = {'command': command, 'limit_seconds': seconds, 'elapsed_seconds': time.monotonic() - start,
               'exit_code': code, 'timed_out': timed_out,
               'stdout_sha256': sha(str(stem) + '.stdout'), 'stderr_sha256': sha(str(stem) + '.stderr')}
    write(str(stem) + '.process.json', receipt)
    return receipt


def inventory(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or any(r['backend_name'] != 'Vulkan0' or r['test_mode'] != 'support' for r in rows):
        raise ValueError('missing or unexpected support inventory')
    return rows


def verify(text, operation, expected=None, witnessed=(), allow_all_unsupported=False):
    cases = collections.Counter()
    passed = collections.Counter()
    unsupported = collections.Counter()
    summaries = []
    errors = []
    for line in ANSI.sub('', text).splitlines():
        match = re.fullmatch(r'  ([A-Z0-9_]+)\((.*)\): (.*)', line)
        if match:
            op, params, outcome = match.groups()
            if op != operation:
                errors.append('unexpected operation: ' + op)
            cases[params] += 1
            if outcome.strip() == 'OK':
                passed[params] += 1
            elif re.fullmatch(r'not supported \[Vulkan0\]\s*', outcome):
                unsupported[params] += 1
            else:
                errors.append('nonpassing or incomplete outcome: ' + params)
        match = re.fullmatch(r'\s*(\d+)/(\d+) tests passed\s*', line)
        if match:
            summaries.append(tuple(map(int, match.groups())))
    n_pass = sum(passed.values())
    if summaries != [(n_pass, n_pass)]:
        errors.append('missing, repeated, or inconsistent numerical summary')
    if not cases or cases != passed + unsupported:
        errors.append('empty or incomplete terminal case accounting')
    if expected is not None and cases != expected:
        errors.append('selected occurrence inventory mismatch')
    if n_pass == 0 and not (allow_all_unsupported and expected is not None and unsupported == expected):
        errors.append('no expected numerical execution')
    for params in witnessed:
        if passed[params] != (expected[params] if expected is not None else cases[params]) or not passed[params]:
            errors.append('previously passing case missing or unsupported: ' + params)
    if expected is None and operation == 'ISTFT' and (unsupported or n_pass < 4):
        errors.append('ISTFT requires all four existing cases supported and passing')
    return {'valid': not errors, 'errors': errors, 'numerical_passes': n_pass,
            'explicit_unsupported': sum(unsupported.values()), 'accounted_occurrences': sum(cases.values()),
            'summary': summaries, 'witnessed_cases_required': len(witnessed)}


def groups_from(rows):
    counter = collections.Counter((r['op_name'], r['op_params']) for r in rows)
    groups = []
    for lane in ['broad', 'custom']:
        for (op, params), count in counter.items():
            if op not in (['MUL_MAT', 'FLASH_ATTN_EXT'] if lane == 'broad' else ['MUL_MAT']):
                continue
            groups.append({'lane': lane, 'operation': op, 'params': params, 'count': count})
    groups.sort(key=lambda g: (g['lane'], g['operation'], g['params']))
    shards = [groups[index::64] for index in range(64)]
    assert sum(g['count'] for shard in shards for g in shard) == sum(g['count'] for g in groups)
    return shards


def weight(group):
    fields = dict(re.findall(r'(\w+)=(\[[^]]*\]|[^,]+)', group['params']))
    num = lambda key: int(fields[key])
    prod = lambda key: math.prod(map(int, fields[key].strip('[]').split(',')))
    if group['operation'] == 'MUL_MAT':
        result = num('m') * num('n') * num('k') * prod('bs') * prod('nr')
    else:
        result = (num('hsk') + num('hsv')) * num('nh') * num('kv') * num('nb') * prod('nr23')
    return result * group['count']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['pilot', 'full'], required=True)
    parser.add_argument('--lane', choices=list(LANES))
    parser.add_argument('--witnesses', default='scripts/ci-vulkan-pilot-witnesses.json')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    binary = str(Path(args.binary).resolve())
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {'head': source, 'qualified_source': SOURCE, 'binary_sha256': sha(binary),
                'mode': args.mode, 'complete_suite': False, 'invocations': [],
                'runtime_files': {str(p.name): sha(p) for p in sorted(Path(binary).parent.iterdir())
                                  if p.is_file() and ('.so' in p.name or '.dylib' in p.name)}}
    write(output / 'receipt.json', manifest)
    if args.mode == 'full':
        if args.lane is None:
            parser.error('--lane is required for full mode')
        selections = [{'lane': args.lane, 'operation': op, 'label': op} for op in LANES[args.lane]]
    else:
        # This pilot measures the already qualified source. Only its workflow,
        # accounting script, and observed-case witnesses may differ from that pin.
        subprocess.run(['git', 'diff', '--exit-code', SOURCE, '--', '.',
                        ':!.github/workflows/eliza-vulkan-validation.yml',
                        ':!scripts/ci-vulkan-partitions.py',
                        ':!scripts/ci-vulkan-pilot-witnesses.json'], check=True)
        witnesses = json.loads(Path(args.witnesses).read_text())
        if witnesses['source'] != SOURCE:
            raise ValueError('witness source mismatch')
        probe = run([binary, 'support', '-b', 'Vulkan0', '-o', 'MUL_MAT,FLASH_ATTN_EXT', '--output', 'csv'], output / 'inventory', 60)
        if probe['exit_code']:
            raise RuntimeError('inventory process failed')
        rows = inventory((output / 'inventory.stdout').read_text())
        identities = sorted((r['op_name'], r['op_params']) for r in rows)
        identity_sha = hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()
        if identity_sha != witnesses['inventory_identity_sha256']:
            raise ValueError('complete source inventory changed')
        shards = groups_from(rows)
        write(output / 'complete-partition-plan.json', shards)
        selections = []
        for lane, op in [('custom', 'MUL_MAT'), ('broad', 'FLASH_ATTN_EXT')]:
            candidates = [[g for g in shard if g['lane'] == lane and g['operation'] == op] for shard in shards]
            heavy = max(range(64), key=lambda index: sum(weight(g) for g in candidates[index]))
            if heavy == 0:
                raise ValueError('representative and heavy partitions unexpectedly coincide')
            for label, index in [('representative', 0), ('estimated-heavy', heavy)]:
                selected = candidates[index]
                expected = {g['params']: g['count'] for g in selected}
                known = [p for p in expected if p in witnesses['passing'][op]]
                if not known:
                    raise ValueError('pilot partition has no independently witnessed numerical case')
                selections.append({'lane': lane, 'operation': op, 'label': op + '-' + label,
                                   'partition': index, 'expected': expected, 'witnessed': known,
                                   'weight_proxy': sum(weight(g) for g in selected)})
        # The first hosted pilot timed out after 78 completed outcomes in this
        # selection. Preserve all 80 occurrences while isolating the two that
        # had no terminal outcome; no selected identity may disappear.
        representative = next(s for s in selections if s['label'] == 'FLASH_ATTN_EXT-representative')
        unresolved = ['hsk=576,hsv=512,nh=1,nr23=[1,1],kv=1024,nb=32,mask=0,sinks=1,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,type_V=f16,permute=[0,1,2,3]', 'hsk=576,hsv=512,nh=4,nr23=[1,1],kv=512,nb=3,mask=1,sinks=1,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,type_V=f16,permute=[0,1,2,3]']
        original = collections.Counter(representative['expected'])
        if len(unresolved) != 2 or any(original[p] != 1 for p in unresolved):
            raise ValueError('unresolved case identities changed')
        completed = sorted(set(original) - set(unresolved))
        subsets = [completed[0::2], completed[1::2], [unresolved[0]], [unresolved[1]]]
        subdivisions = []
        union = collections.Counter()
        for index, params in enumerate(subsets):
            expected = {p: original[p] for p in params}
            union.update(expected)
            prior_witnesses = [p for p in representative['witnessed'] if p in expected]
            required_passes = prior_witnesses if index < 2 else params
            subdivisions.append({'lane': 'broad', 'operation': 'FLASH_ATTN_EXT',
                                 'label': 'FLASH_ATTN_EXT-subdivision-' + str(index),
                                 'expected': expected, 'witnessed': required_passes,
                                 'prior_witnesses': prior_witnesses,
                                 'isolated_unresolved': index >= 2})
        if union != original or sum(original.values()) != 80:
            raise ValueError('subdivision changed the original selected inventory')
        if sum(len(s['prior_witnesses']) for s in subdivisions) != len(representative['witnessed']):
            raise ValueError('subdivision dropped a prior witness')
        selections = subdivisions
        manifest['followup_of_run'] = 35467120771
        manifest['original_selected_occurrences'] = sum(original.values())
        manifest['inventory_identity_sha256'] = identity_sha
        manifest['planned_original_occurrences'] = sum(g['count'] for shard in shards for g in shard)
        manifest['pilot_only'] = True
    write(output / 'selections.json', selections)
    for selection in selections:
        op = selection['operation']
        command = [binary, '-b', 'Vulkan0', '-o', op]
        expected = selection.get('expected')
        if expected is not None:
            command += ['-p', '^(' + '|'.join(re.escape(p) for p in sorted(expected)) + ')$']
        stem = output / selection['label']
        receipt = run(command, stem, 300 if selection['lane'] == 'custom' else 600)
        accounting = verify(Path(str(stem) + '.stdout').read_text(), op,
                            collections.Counter(expected) if expected is not None else None,
                            selection.get('witnessed', []))
        receipt.update({'selection': selection, 'accounting': accounting,
                        'valid': receipt['exit_code'] == 0 and accounting['valid']})
        manifest['invocations'].append(receipt)
        write(output / 'receipt.json', manifest)
        print(json.dumps({'label': selection['label'], 'exit_code': receipt['exit_code'], 'accounting': accounting}), flush=True)
    manifest['valid'] = all(r['valid'] for r in manifest['invocations'])
    manifest['complete_requested_invocations'] = manifest['valid']
    write(output / 'receipt.json', manifest)
    return 0 if manifest['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
