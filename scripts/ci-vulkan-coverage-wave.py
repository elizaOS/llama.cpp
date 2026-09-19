#!/usr/bin/env python3
"""Run bounded Vulkan coverage workers and independently account for every case.

Completed pilot receipts remain source-bound evidence. A worker checkpoint is
not suite success: global acceptance requires the complete original multiset.
"""
import argparse
import collections
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import time

spec = importlib.util.spec_from_file_location('partitions', Path(__file__).with_name('ci-vulkan-partitions.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
SOURCE = helper.SOURCE


def write(path, value):
    helper.write(path, value)


def counter(selection):
    return collections.Counter({(selection['lane'], selection['operation'], p): n
                                for p, n in selection['expected'].items()})


def verify_invocation(invocation, folder):
    selection = invocation['selection']
    label = selection['label']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('unsafe invocation label')
    stdout = folder / (label + '.stdout')
    stderr = folder / (label + '.stderr')
    if helper.sha(stdout) != invocation['stdout_sha256'] or helper.sha(stderr) != invocation['stderr_sha256']:
        raise ValueError('invocation output hash mismatch')
    accounting = helper.verify(stdout.read_text(), selection['operation'],
                               collections.Counter(selection['expected']), selection.get('witnessed', []),
                               allow_all_unsupported=True)
    if invocation['exit_code'] != 0 or invocation['timed_out'] or not accounting['valid']:
        raise ValueError('nonterminal or nonpassing invocation')
    if json.loads(json.dumps(accounting)) != invocation['accounting']:
        raise ValueError('stored accounting mismatch')
    return counter(selection)


def original_inventory(rows):
    result = collections.Counter()
    for shard in helper.groups_from(rows):
        for group in shard:
            result[(group['lane'], group['operation'], group['params'])] += group['count']
    return result


def libraries(binary):
    result = subprocess.check_output(['ldd', str(binary)], text=True)
    result += subprocess.check_output(['ldd', '/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so'], text=True)
    if 'not found' in result:
        raise ValueError('unresolved runtime dependency')
    paths = re.findall(r'=> (/\S+)', result)
    paths += re.findall(r'^\s*(/\S+) \(', result, re.M)
    paths += ['/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so']
    return result, {Path(p).name: helper.sha(p) for p in sorted(set(paths))}


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(Path(args.inputs).read_text())
    binary = Path(args.binary).resolve()
    subprocess.run(['git', 'diff', '--exit-code', SOURCE, '--', '.',
                    ':!.github/workflows/eliza-vulkan-validation.yml',
                    ':!scripts/ci-vulkan-partitions.py', ':!scripts/ci-vulkan-pilot-witnesses.json',
                    ':!scripts/ci-vulkan-coverage-wave.py', ':!scripts/ci-vulkan-wave-inputs.json'], check=True)
    completed = collections.Counter()
    prior_passes = collections.defaultdict(set)
    for prior in inputs['prior']:
        folder = Path(args.prior) / prior['directory'] / 'vulkan-validation/pilot'
        if helper.sha(folder / 'receipt.json') != prior['receipt_sha256']:
            raise ValueError('prior receipt binding mismatch')
        receipt = json.loads((folder / 'receipt.json').read_text())
        if receipt['qualified_source'] != SOURCE or receipt['head'] != prior['head']:
            raise ValueError('prior source mismatch')
        for invocation in receipt['invocations']:
            if not invocation['valid']:
                continue
            completed.update(verify_invocation(invocation, folder))
            op = invocation['selection']['operation']
            for line in helper.ANSI.sub('', (folder / (invocation['selection']['label'] + '.stdout')).read_text()).splitlines():
                match = re.fullmatch(r'  ' + op + r'\((.*)\): OK\s*', line)
                if match:
                    prior_passes[op].add(match[1])
    probe = helper.run([str(binary), 'support', '-b', 'Vulkan0', '-o', 'MUL_MAT,FLASH_ATTN_EXT', '--output', 'csv'], output / 'inventory', 60)
    if probe['exit_code']:
        raise ValueError('inventory failed')
    rows = helper.inventory((output / 'inventory.stdout').read_text())
    import hashlib
    identity = hashlib.sha256(json.dumps(sorted((r['op_name'], r['op_params']) for r in rows), separators=(',', ':')).encode()).hexdigest()
    if identity != inputs['inventory_identity_sha256']:
        raise ValueError('inventory identity mismatch')
    original = original_inventory(rows)
    remaining = original - completed
    if completed - original or sum(original.values()) != 7922 or sum(completed.values()) != 203:
        raise ValueError('prior coverage multiset mismatch')
    if completed != collections.Counter({tuple(r[:3]): r[3] for r in inputs['completed']}):
        raise ValueError('pinned completed multiset mismatch')
    planned = collections.Counter()
    for partition in inputs['partitions']:
        planned.update(counter(partition))
        if partition['witnessed'] != sorted(set(partition['expected']) & prior_passes[partition['operation']]):
            raise ValueError('prior numerical witness mismatch')
    if planned != remaining or len(inputs['partitions']) != 375:
        raise ValueError('remaining partition multiset mismatch')
    ldd, runtime = libraries(binary)
    (output / 'ldd.txt').write_text(ldd)
    write(output / 'build.json', {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'source': SOURCE, 'input_sha256': helper.sha(args.inputs), 'inventory_identity_sha256': identity,
          'binary_sha256': helper.sha(binary), 'runtime_libraries': runtime,
          'payload_files': {p.name: helper.sha(p) for p in sorted(binary.parent.iterdir()) if p.is_file()},
          'partitions': inputs['partitions'], 'completed': [list(k) + [n] for k, n in sorted(completed.items())],
          'original': [list(k) + [n] for k, n in sorted(original.items())]})


def worker(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    build = json.loads(Path(args.build).read_text())
    if build['source'] != SOURCE or build['head'] != subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip():
        raise ValueError('worker source mismatch')
    binary = Path(args.binary).resolve()
    for name, digest in build['payload_files'].items():
        if Path(name).name != name or helper.sha(binary.parent / name) != digest:
            raise ValueError('binary payload mismatch')
    ldd, runtime = libraries(binary)
    (output / 'ldd.txt').write_text(ldd)
    if runtime != build['runtime_libraries']:
        raise ValueError('runtime library mismatch')
    assigned = build['partitions'][args.worker::4]
    receipt = {'head': build['head'], 'source': SOURCE, 'worker': args.worker,
               'build_sha256': helper.sha(args.build), 'complete_suite': False,
               'assigned_ids': [p['id'] for p in assigned], 'invocations': [], 'pending_ids': [p['id'] for p in assigned]}
    write(output / 'receipt.json', receipt)
    started = time.monotonic()
    queues = [[p for p in assigned if (p['lane'], p['operation']) == key]
              for key in [('broad', 'FLASH_ATTN_EXT'), ('broad', 'MUL_MAT'), ('custom', 'MUL_MAT')]]
    execution = [p for row in itertools.zip_longest(*queues) for p in row if p is not None]
    for partition in execution:
        # Reserve a complete original deadline plus termination grace. Never
        # shorten an invocation to fill the worker's remaining time budget.
        limit = partition['limit_seconds']
        if time.monotonic() - started + limit + 5 > 48 * 60:
            break
        selection = dict(partition, label='partition-' + str(partition['id']))
        command = [str(binary), '-b', 'Vulkan0', '-o', partition['operation'], '-p',
                   '^(' + '|'.join(re.escape(p) for p in sorted(partition['expected'])) + ')$']
        inv = helper.run(command, output / selection['label'], limit)
        inv['selection'] = selection
        inv['accounting'] = helper.verify((output / (selection['label'] + '.stdout')).read_text(), selection['operation'],
                                         collections.Counter(selection['expected']), selection['witnessed'],
                                         allow_all_unsupported=True)
        inv['valid'] = inv['exit_code'] == 0 and inv['accounting']['valid']
        receipt['invocations'].append(inv)
        if inv['valid']:
            receipt['pending_ids'].remove(partition['id'])
        write(output / 'receipt.json', receipt)
        print(json.dumps({'partition': partition['id'], 'exit': inv['exit_code'], 'accounting': inv['accounting']}), flush=True)
        if not inv['valid']:
            break
    receipt['worker_finished'] = True
    receipt['elapsed_seconds'] = time.monotonic() - started
    write(output / 'receipt.json', receipt)
    return 0 if all(i['valid'] for i in receipt['invocations']) else 1


def aggregate(args):
    build = json.loads(Path(args.build).read_text())
    inputs = json.loads(Path(args.inputs).read_text())
    if build['source'] != SOURCE or build['input_sha256'] != helper.sha(args.inputs):
        raise ValueError('aggregate source/input binding mismatch')
    if build['partitions'] != inputs['partitions'] or build['completed'] != inputs['completed']:
        raise ValueError('aggregate build/pinned ledger mismatch')
    seen = collections.Counter({tuple(row[:3]): row[3] for row in inputs['completed']})
    original = seen.copy()
    for partition in inputs['partitions']:
        original.update(counter(partition))
    if original != collections.Counter({tuple(row[:3]): row[3] for row in build['original']}):
        raise ValueError('aggregate original inventory mismatch')
    completed_ids = set()
    errors = []
    workers = set()
    for receipt_path in sorted(Path(args.receipts).glob('worker-*/receipt.json')):
        receipt = json.loads(receipt_path.read_text())
        worker_id = receipt['worker']
        if worker_id in workers or worker_id not in range(4) or receipt['build_sha256'] != helper.sha(args.build):
            errors.append('duplicate worker or build mismatch')
            continue
        workers.add(worker_id)
        if receipt['head'] != build['head'] or receipt['source'] != SOURCE or not receipt.get('worker_finished'):
            errors.append('worker source or terminal marker mismatch')
        assigned = build['partitions'][worker_id::4]
        if receipt['assigned_ids'] != [p['id'] for p in assigned]:
            errors.append('worker assignment mismatch')
        for inv in receipt['invocations']:
            selection = inv['selection']
            ident = selection['id']
            expected = build['partitions'][ident] if isinstance(ident, int) and 0 <= ident < len(build['partitions']) else None
            if ident in completed_ids or expected is None or ident % 4 != worker_id or any(selection[k] != expected[k] for k in expected):
                errors.append('duplicate partition, lane, or expected inventory mismatch')
                continue
            try:
                verified = verify_invocation(inv, receipt_path.parent)
            except (ValueError, FileNotFoundError) as error:
                errors.append(str(error))
                continue
            completed_ids.add(ident)
            seen.update(verified)
        expected_pending = [p['id'] for p in assigned if p['id'] not in completed_ids]
        if receipt['pending_ids'] != expected_pending:
            errors.append('worker pending ledger mismatch')
    missing = original - seen
    if workers != set(range(4)):
        errors.append('missing worker receipt')
    if seen - original:
        errors.append('unexpected or duplicate case occurrences')
    if missing:
        errors.append('full original coverage incomplete')
    write(args.output, {'source': SOURCE, 'head': build['head'], 'complete_suite': False,
          'complete_original_three_invocations': not errors, 'errors': errors,
          'covered_occurrences': sum(seen.values()), 'original_occurrences': sum(original.values()),
          'missing': [list(k) + [n] for k, n in sorted(missing.items())],
          'completed_partition_ids': sorted(completed_ids)})
    return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'worker', 'aggregate'])
    parser.add_argument('--binary')
    parser.add_argument('--output', required=True)
    parser.add_argument('--inputs', default='scripts/ci-vulkan-wave-inputs.json')
    parser.add_argument('--prior')
    parser.add_argument('--build')
    parser.add_argument('--worker', type=int, choices=range(4))
    parser.add_argument('--receipts')
    args = parser.parse_args()
    return {'prepare': prepare, 'worker': worker, 'aggregate': aggregate}[args.mode](args)


if __name__ == '__main__':
    raise SystemExit(main())
