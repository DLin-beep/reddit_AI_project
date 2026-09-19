#!/usr/bin/env python3
"""Continue unfinished vMF fits without a total EM iteration cutoff.

Original results are immutable. Scheduler limits pause a checkpointed task;
only the unchanged numerical convergence criterion can complete it.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import sys
import time
import uuid
from collections import Counter
from itertools import combinations
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run_semantic_threefold_pilot as pilot
from semantic_vmf_continuation import continue_vmf

ARTIFACTS = ('fit.json', 'model.npz', 'predictions.npz', 'review_assignments_restricted.csv')
META = ('config.json', 'inputs.json', 'tasks.json', 'fold_assignments_restricted.csv')
SOURCES = ('run_vmf_convergence_extension.py', 'semantic_vmf_continuation.py',
           'run_semantic_threefold_pilot.py', 'run_semantic_region_pilot.py',
           'semantic_region_directional.py', 'semantic_region_density.py',
           'semantic_threefold_density.py', 'semantic_threefold_report.py')


def read_json(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {name: pilot.sha256(Path(__file__).with_name(name)) for name in SOURCES}


def runtime_identity():
    return {'python': sys.version, **{name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')}}


def task_path(output, task_id):
    return Path(output) / 'tasks' / f'task_{task_id:03d}'


def load_model(directory):
    fit = read_json(Path(directory) / 'fit.json')
    with np.load(Path(directory) / 'model.npz', allow_pickle=False) as data:
        model = {name: data[name].copy() for name in data.files
                 if name not in ('input_fingerprint', 'training_ids')}
    model.update(fit['model_scalars'])
    return model


def convergence_evidence(model, n, tolerance):
    history = np.asarray(model.get('log_likelihood_history', []), dtype=float)
    if history.ndim != 1 or len(history) != int(model['iterations']) + 1 or not np.isfinite(history).all():
        raise ValueError('Invalid saved likelihood history')
    if len(history) < 2 or not np.isclose(history[-1], model['log_likelihood'], rtol=1e-12, atol=1e-7):
        raise ValueError('Saved likelihood and final history value disagree')
    improvements = np.diff(history) / n
    if np.any(improvements < -1e-7):
        raise ValueError('Saved EM history contains a material likelihood decrease')
    final = float(improvements[-1])
    actual = final <= tolerance
    if bool(model['converged']) != actual:
        raise ValueError('Saved convergence flag disagrees with numerical stopping criterion')
    return {'converged': actual, 'iterations': int(model['iterations']),
            'final_mean_log_likelihood_improvement': final, 'tolerance': tolerance}


def source_task(source, spec, source_inputs, config_hash):
    directory = task_path(source, spec['task_id'])
    if not pilot.validate_completed_task(directory, spec, source_inputs):
        raise ValueError('Original experiment must have all task artifacts finished')
    status, fit = read_json(directory / 'status.json'), read_json(directory / 'fit.json')
    if status.get('config_sha256') != config_hash or fit.get('config_sha256') != config_hash:
        raise ValueError('Original task configuration hash mismatch')
    if fit.get('input_fingerprint') != source_inputs['input_fingerprint']:
        raise ValueError('Original task input fingerprint mismatch')
    return directory, status, fit


def bind_model(path, source_path, fingerprint):
    """Change only provenance metadata; all fitted numerical arrays are retained."""
    with np.load(source_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays['input_fingerprint'] = np.asarray(fingerprint)
    np.savez_compressed(path, **arrays)


def complete_status(directory, spec, summary, **extra):
    status = dict(status='complete', task_id=spec['task_id'], fit_id=spec['fit_id'],
                  input_fingerprint=summary['input_fingerprint'], config_sha256=summary['config_sha256'],
                  completed_utc=pilot.now(), artifact_sha256={name: pilot.sha256(directory / name) for name in ARTIFACTS})
    status.update(extra)
    pilot.write_json(directory / 'status.json', status)


def prepare(source_results, data_root, output, *, chunk_iterations=200, checkpoint_every=10):
    source, output = Path(source_results).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Extension output must be separate from the original result directory')
    pilot.positive_integer(chunk_iterations, 'chunk_iterations')
    pilot.positive_integer(checkpoint_every, 'checkpoint_every')
    if output.exists() and (output / 'extension.json').exists():
        config, inputs, summary, extension = load_run(source, data_root, output)
        if config['optimization']['chunk_iterations'] != chunk_iterations or config['optimization']['checkpoint_every'] != checkpoint_every:
            raise ValueError('Prepared checkpoint policy differs from requested policy')
        return summary
    source_config, source_inputs = read_json(source / 'config.json'), read_json(source / 'inputs.json')
    inputs = pilot.load_inputs(source_config, data_root)
    specs = read_json(source / 'tasks.json')
    if specs != pilot.planned_tasks(source_config):
        raise ValueError('Original task plan differs from original configuration')
    if pilot.read_csv(source / 'fold_assignments_restricted.csv') != [
            {key: str(value) for key, value in row.items()} for row in pilot.fold_rows(inputs)]:
        raise ValueError('Original fold assignments differ from authenticated corpus')
    original_hash = pilot.digest(source_config)
    if source_inputs['config_sha256'] != original_hash or source_inputs['input_sha256'] != inputs['input_sha256']:
        raise ValueError('Original committed inputs differ from authenticated inputs')
    source_commit = dict(source_inputs)
    fingerprint = source_commit.pop('input_fingerprint')
    if pilot.digest(source_commit) != fingerprint:
        raise ValueError('Original input fingerprint is invalid')
    if source_inputs['runtime'] != runtime_identity():
        raise ValueError('Continuation runtime differs from the original; preserve the pinned numerical environment')
    current_sources = source_hashes()
    for name, expected in source_inputs['code_sha256'].items():
        if current_sources.get(name) != expected:
            raise ValueError(f'Original implementation changed: {name}')
    identities, pending = {}, []
    for spec in specs:
        directory, status, fit = source_task(source, spec, source_inputs, original_hash)
        model = load_model(directory)
        evidence = None
        if spec['family'] == 'vmf':
            evidence = convergence_evidence(model, sum(len(inputs['folds'][f]) for f in spec['train_folds']),
                                            source_config['models']['tolerance'])
            if not evidence['converged']:
                pending.append(spec['task_id'])
        elif spec['family'] == 'spherical_kmeans' and not model.get('converged'):
            raise ValueError('An unfinished spherical k-means fit requires a separate continuation implementation')
        identities[str(spec['task_id'])] = {'status_sha256': pilot.sha256(directory / 'status.json'),
                                          'artifact_sha256': status['artifact_sha256'], 'convergence': evidence}
    source_manifest = {'source_results': str(source), 'metadata_sha256': {name: pilot.sha256(source / name) for name in META},
                       'tasks': identities, 'input_fingerprint': fingerprint, 'config_sha256': original_hash}
    config = copy.deepcopy(source_config)
    config['name'] = 'semantic_threefold_convergence_extension'
    config['optimization'] = {'total_em_iteration_limit': None, 'chunk_iterations': chunk_iterations,
                              'checkpoint_every': checkpoint_every, 'original_initialization_max_iter': source_config['models']['max_iter'],
                              'models_max_iter_scope': 'Historical original fit and initialization only; not an EM continuation cap.',
                              'tolerance': source_config['models']['tolerance'], 'kappa_max': source_config['models']['vmf_kappa_max'],
                              'completion': 'Numerical convergence only; wall deadlines leave resumable incomplete tasks.',
                              'source_experiment_sha256': pilot.digest(source_manifest)}
    summary = copy.deepcopy(source_inputs)
    summary.update(config_sha256=pilot.digest(config), code_sha256=source_hashes(),
                   source_input_fingerprint=fingerprint, source_experiment_sha256=pilot.digest(source_manifest))
    summary.pop('input_fingerprint')
    summary['input_fingerprint'] = pilot.digest(summary)
    extension = {'source': source_manifest, 'pending_task_ids': pending,
                 'carried_forward_task_ids': [spec['task_id'] for spec in specs if spec['task_id'] not in pending],
                 'input_fingerprint': summary['input_fingerprint'], 'created_utc': pilot.now()}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + '.prepare.' + uuid.uuid4().hex)
    if output.exists():
        raise FileExistsError('Extension output already exists without a complete preparation commit')
    staging.mkdir()
    try:
        for name in ('tasks', 'locks', 'checkpoints'):
            (staging / name).mkdir()
        pilot.write_json(staging / 'config.json', config)
        pilot.write_json(staging / 'inputs.json', summary)
        pilot.write_json(staging / 'tasks.json', specs)
        pilot.write_json(staging / 'extension.json', extension)
        shutil.copyfile(source / 'fold_assignments_restricted.csv', staging / 'fold_assignments_restricted.csv')
        for spec in specs:
            directory = task_path(staging, spec['task_id'])
            directory.mkdir()
            provenance = {'source_task_id': spec['task_id'], 'source_results': str(source),
                          'source_artifact_sha256': identities[str(spec['task_id'])]['artifact_sha256'],
                          'source_input_fingerprint': fingerprint}
            if spec['task_id'] in pending:
                pilot.write_json(directory / 'status.json', dict(status='pending_continuation', task_id=spec['task_id'],
                                 fit_id=spec['fit_id'], input_fingerprint=summary['input_fingerprint'],
                                 config_sha256=summary['config_sha256'], provenance=provenance))
                continue
            original = task_path(source, spec['task_id'])
            for name in ('predictions.npz', 'review_assignments_restricted.csv'):
                shutil.copyfile(original / name, directory / name)
            bind_model(directory / 'model.npz', original / 'model.npz', summary['input_fingerprint'])
            fit = read_json(original / 'fit.json')
            fit.update(input_fingerprint=summary['input_fingerprint'], config_sha256=summary['config_sha256'],
                       continuation=dict(provenance, action='carried_forward_numerically_unchanged', added_em_iterations=0))
            pilot.write_json(directory / 'fit.json', fit)
            complete_status(directory, spec, summary, action='carried_forward_numerically_unchanged')
        pilot.write_json(staging / 'status.json', dict(status='prepared', planned_fits=len(specs),
                         completed_fits=len(specs) - len(pending), pending_task_ids=pending,
                         input_fingerprint=summary['input_fingerprint'], prepared_utc=pilot.now()))
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return summary


def load_run(source_results, data_root, output):
    source, output = Path(source_results).resolve(), Path(output).resolve()
    extension = read_json(output / 'extension.json')
    config, summary = read_json(output / 'config.json'), read_json(output / 'inputs.json')
    if extension['source']['source_results'] != str(source):
        raise ValueError('Original source results path changed')
    if pilot.digest(extension['source']) != config['optimization']['source_experiment_sha256']:
        raise ValueError('Source experiment provenance was modified')
    if summary['config_sha256'] != pilot.digest(config) or summary['code_sha256'] != source_hashes():
        raise ValueError('Extension configuration or source code changed; cannot resume')
    if summary['runtime'] != runtime_identity():
        raise ValueError('Continuation numerical runtime changed; cannot resume')
    committed = dict(summary)
    fingerprint = committed.pop('input_fingerprint')
    if pilot.digest(committed) != fingerprint or extension['input_fingerprint'] != fingerprint:
        raise ValueError('Extension input fingerprint is invalid')
    for name, expected in extension['source']['metadata_sha256'].items():
        if pilot.sha256(source / name) != expected:
            raise ValueError('Original experiment metadata changed')
    if read_json(output / 'tasks.json') != read_json(source / 'tasks.json'):
        raise ValueError('Extension task plan was modified')
    tasks = read_json(output / 'tasks.json')
    expected_pending = [spec['task_id'] for spec in tasks if spec['family'] == 'vmf'
                        and not extension['source']['tasks'][str(spec['task_id'])]['convergence']['converged']]
    if extension['pending_task_ids'] != expected_pending or extension['carried_forward_task_ids'] != [
            spec['task_id'] for spec in tasks if spec['task_id'] not in expected_pending]:
        raise ValueError('Extension continuation task inventory was modified')
    if pilot.sha256(output / 'fold_assignments_restricted.csv') != extension['source']['metadata_sha256']['fold_assignments_restricted.csv']:
        raise ValueError('Extension fold assignments were modified')
    inputs = pilot.load_inputs(config, data_root)
    if inputs['input_sha256'] != summary['input_sha256']:
        raise ValueError('Extension data input identity changed')
    return config, inputs, summary, extension


def checkpoint_binding(spec, summary, extension):
    return {'task_id': spec['task_id'], 'input_fingerprint': summary['input_fingerprint'],
            'config_sha256': summary['config_sha256'], 'code_sha256': summary['code_sha256'],
            'source_task': extension['source']['tasks'][str(spec['task_id'])]}


def save_checkpoint(directory, model, training_ids, binding, elapsed_seconds):
    directory.mkdir(parents=True, exist_ok=True)
    generation = f"state_{int(model['iterations']):08d}_{uuid.uuid4().hex}"
    array_path = directory / (generation + '.npz')
    arrays = {name: value for name, value in model.items() if isinstance(value, np.ndarray)}
    arrays['training_ids'] = np.asarray(training_ids)
    # An immutable payload is durable before the atomic commit pointer changes.
    with array_path.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {'binding': binding, 'model_file': array_path.name, 'model_sha256': pilot.sha256(array_path),
                'model_scalars': pilot.scalar_model_metadata(model), 'elapsed_seconds': elapsed_seconds,
                'checkpoint_utc': pilot.now()}
    pilot.write_json(directory / 'checkpoint.json', manifest)
    # Keep the committed payload and one predecessor; never remove original data.
    old = sorted((path for path in directory.glob('state_*.npz') if path != array_path), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in old[1:]:
        path.unlink()
    return manifest


def load_checkpoint(directory, training_ids, binding):
    path = directory / 'checkpoint.json'
    if not path.exists():
        return None
    manifest = read_json(path)
    if manifest['binding'] != binding:
        raise ValueError('Checkpoint belongs to changed inputs, source code, or task')
    name = manifest['model_file']
    if Path(name).name != name or not name.startswith('state_') or not name.endswith('.npz'):
        raise ValueError('Invalid checkpoint payload path')
    payload = directory / name
    if payload.is_symlink() or pilot.sha256(payload) != manifest['model_sha256']:
        raise ValueError('Checkpoint payload hash mismatch')
    with np.load(payload, allow_pickle=False) as data:
        if data['training_ids'].tolist() != training_ids:
            raise ValueError('Checkpoint training post order differs from the original fit')
        model = {name: data[name].copy() for name in data.files if name != 'training_ids'}
    model.update(manifest['model_scalars'])
    return model, float(manifest['elapsed_seconds'])


def write_final_artifacts(config, inputs, spec, task_dir, summary, model, vectors, verified,
                          original_fit, training_ids, elapsed_seconds, extension):
    arrays = {'training_ids': np.asarray(training_ids), 'training_labels': np.asarray(model['labels'], dtype=np.int32)}
    summaries, labels_by_id = [], dict(zip(training_ids, model['labels'].tolist()))
    for fold in spec['eval_folds']:
        ids = inputs['folds'][fold]
        E = np.asarray([vectors[aid] for aid in ids], dtype=np.float64)
        labels, confidence, probabilities, log_density = pilot.predict('vmf', E, model, config['resources']['threads'])
        arrays.update({f'evaluation_ids_{fold}': np.asarray(ids), f'evaluation_labels_{fold}': labels.astype(np.int32),
                       f'evaluation_centrality_{fold}': pilot.centrality(E, labels),
                       f'evaluation_confidence_{fold}': confidence.astype(np.float32),
                       f'evaluation_memberships_{fold}': probabilities.astype(np.float32),
                       f'evaluation_log_density_{fold}': log_density})
        labels_by_id.update(zip(ids, labels.tolist()))
        row = pilot.evaluation_summary(spec, fold, model, labels, confidence, log_density, config['models'], len(training_ids))
        row['seconds'] = original_fit['fit_seconds'] + elapsed_seconds
        summaries.append(row)
        del E
    review_ids, training_set = sorted(inputs['reviews']), set(training_ids)
    fold_by_id = {aid: fold for fold, ids in inputs['folds'].items() for aid in ids}
    arrays['review_ids'] = np.asarray(review_ids)
    arrays['review_labels'] = np.asarray([labels_by_id[aid] for aid in review_ids], dtype=np.int32)
    arrays['review_used_in_fit'] = np.asarray([aid in training_set for aid in review_ids], dtype=bool)
    review_rows = [dict(inputs['reviews'][aid], fit=spec['fit_id'], fit_id=spec['fit_id'], annotation_id=aid,
                       historical_role=inputs['roles'][aid]['sample_role'], used_in_fit=int(aid in training_set),
                       fold=fold_by_id[aid], region=int(labels_by_id[aid]),
                       assignment_basis='fitted_training_label' if aid in training_set else 'out_of_training_prediction') for aid in review_ids]
    for row in summaries:
        row['human_review_coverage'] = 1.0 if review_ids else None
    model_arrays = {name: value for name, value in model.items() if isinstance(value, np.ndarray)}
    model_arrays.update(training_ids=np.asarray(training_ids), input_fingerprint=np.asarray(summary['input_fingerprint']))
    np.savez_compressed(task_dir / 'model.npz', **model_arrays)
    np.savez_compressed(task_dir / 'predictions.npz', **arrays)
    pilot.write_csv(task_dir / 'review_assignments_restricted.csv', review_rows)
    original_iterations = int(original_fit['model_scalars']['iterations'])
    fit = {'task': spec, 'summaries': summaries, 'model_scalars': pilot.scalar_model_metadata(model),
           'fit_seconds': original_fit['fit_seconds'] + elapsed_seconds, 'input_fingerprint': summary['input_fingerprint'],
           'config_sha256': summary['config_sha256'], 'verified_shard_sha256': verified,
           'training_embedding_dimension': config['input']['dimension'],
           'continuation': {'action': 'continued_original_parameters_to_convergence', 'source_results': extension['source']['source_results'],
                            'source_artifact_sha256': extension['source']['tasks'][str(spec['task_id'])]['artifact_sha256'],
                            'source_input_fingerprint': extension['source']['input_fingerprint'],
                            'original_iterations': original_iterations, 'added_em_iterations': int(model['iterations']) - original_iterations,
                            'continuation_seconds': elapsed_seconds, 'total_em_iteration_limit': None,
                            'timing_scope': 'Continuation seconds include input loading, checkpoint I/O, and prediction setup; not pure EM compute time.',
                            'completion_evidence': convergence_evidence(model, len(training_ids), config['models']['tolerance'])}}
    pilot.write_json(task_dir / 'fit.json', fit)


def run_task(source_results, data_root, output, task_id, *, wall_seconds=6600):
    if not np.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError('wall_seconds must be positive and finite')
    config, inputs, summary, extension = load_run(source_results, data_root, output)
    specs = read_json(Path(output) / 'tasks.json')
    matches = [spec for spec in specs if spec['task_id'] == task_id]
    if len(matches) != 1:
        raise ValueError('Task ID outside original task plan')
    spec, directory = matches[0], task_path(output, task_id)
    with pilot.file_lock(Path(output) / 'locks' / f'task_{task_id:03d}.lock'):
        if pilot.validate_completed_task(directory, spec, summary):
            print(json.dumps({'task_id': task_id, 'status': 'already_complete'}), flush=True)
            return 0
        if task_id not in extension['pending_task_ids'] or spec['family'] != 'vmf':
            raise ValueError('Only originally unfinished vMF tasks may be continued')
        original = task_path(source_results, task_id)
        identity = extension['source']['tasks'][str(task_id)]
        if pilot.sha256(original / 'status.json') != identity['status_sha256']:
            raise ValueError('Original task status changed')
        for name, expected in identity['artifact_sha256'].items():
            if pilot.sha256(original / name) != expected:
                raise ValueError('Original task artifact changed')
        training_ids = [aid for fold in spec['train_folds'] for aid in inputs['folds'][fold]]
        with np.load(original / 'model.npz', allow_pickle=False) as data:
            if data['training_ids'].tolist() != training_ids:
                raise ValueError('Original training order differs from unchanged folds')
        with np.load(original / 'predictions.npz', allow_pickle=False) as data:
            for fold in spec['eval_folds']:
                if data[f'evaluation_ids_{fold}'].tolist() != inputs['folds'][fold]:
                    raise ValueError('Original evaluation order differs from unchanged folds')
        checkpoint_dir = Path(output) / 'checkpoints' / f'task_{task_id:03d}'
        binding = checkpoint_binding(spec, summary, extension)
        checkpoint = load_checkpoint(checkpoint_dir, training_ids, binding)
        model, elapsed_before = checkpoint if checkpoint is not None else (load_model(original), 0.)
        started, stopped = time.monotonic(), [False]
        previous_handlers = {}
        def request_stop(signum, frame):
            stopped[0] = True
        for name in ('SIGTERM', 'SIGUSR1', 'SIGINT'):
            signum = getattr(signal, name, None)
            if signum is not None:
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_stop)
        def should_stop():
            return stopped[0] or time.monotonic() - started >= wall_seconds
        state = {'status': 'running', 'task_id': task_id, 'fit_id': spec['fit_id'],
                 'input_fingerprint': summary['input_fingerprint'], 'config_sha256': summary['config_sha256'],
                 'started_utc': pilot.now(), 'resumed_from_iterations': int(model['iterations'])}
        pilot.write_json(directory / 'status.json', state)
        def checkpoint_model(current):
            elapsed = elapsed_before + time.monotonic() - started
            save_checkpoint(checkpoint_dir, current, training_ids, binding, elapsed)
            state.update(iterations=int(current['iterations']), converged=bool(current['converged']),
                         checkpoint_utc=pilot.now(), continuation_seconds=elapsed)
            pilot.write_json(directory / 'status.json', state)
            print(json.dumps({'task_id': task_id, 'iterations': int(current['iterations']),
                              'converged': bool(current['converged']), 'continuation_seconds': elapsed}), flush=True)
        try:
            ids = [aid for fold in pilot.FOLDS for aid in inputs['folds'][fold]]
            vectors, verified = pilot.load_vectors(inputs['index'], ids, inputs['manifest'], inputs['shard_dir'], config['input']['dimension'])
            X = np.asarray([vectors[aid] for aid in training_ids], dtype=np.float64)
            with threadpool_limits(limits=config['resources']['threads']):
                while True:
                    model = continue_vmf(X, model, tol=config['models']['tolerance'], kappa_max=config['models']['vmf_kappa_max'],
                                         chunk_iterations=config['optimization']['chunk_iterations'], checkpoint_callback=checkpoint_model,
                                         checkpoint_every=config['optimization']['checkpoint_every'], stop_requested=should_stop)
                    if model['converged']:
                        convergence_evidence(model, len(X), config['models']['tolerance'])
                        break
                    if should_stop() or model.get('stop_reason') == 'stop_requested':
                        state.update(status='incomplete', stop_reason='wall_deadline_or_signal', stopped_utc=pilot.now())
                        pilot.write_json(directory / 'status.json', state)
                        return 75
                write_final_artifacts(config, inputs, spec, directory, summary, model, vectors, verified,
                                      read_json(original / 'fit.json'), training_ids,
                                      elapsed_before + time.monotonic() - started, extension)
            complete_status(directory, spec, summary, action='continued_original_parameters_to_convergence',
                            iterations=int(model['iterations']), converged=True)
            print(json.dumps({'task_id': task_id, 'status': 'complete', 'iterations': int(model['iterations']), 'converged': True}), flush=True)
            return 0
        except Exception as exc:
            state.update(status='failed', error_type=type(exc).__name__, error=str(exc), stopped_utc=pilot.now())
            pilot.write_json(directory / 'status.json', state)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def aggregate(source_results, data_root, output):
    config, inputs, summary, extension = load_run(source_results, data_root, output)
    specs, rows, incomplete = read_json(Path(output) / 'tasks.json'), [], []
    with pilot.file_lock(Path(output) / '.aggregate.lock'):
        for spec in specs:
            directory = task_path(output, spec['task_id'])
            if not pilot.validate_completed_task(directory, spec, summary):
                status = read_json(directory / 'status.json')
                incomplete.append(dict(task_id=spec['task_id'], status=status.get('status'), iterations=status.get('iterations')))
                continue
            model = load_model(directory)
            if spec['family'] == 'spherical_kmeans' and not model['converged']:
                raise ValueError('A completed k-means task did not converge')
            if spec['family'] != 'vmf':
                continue
            evidence = convergence_evidence(model, sum(len(inputs['folds'][f]) for f in spec['train_folds']), config['models']['tolerance'])
            if not evidence['converged']:
                raise ValueError('A completed vMF task did not converge')
            original_iterations = extension['source']['tasks'][str(spec['task_id'])]['convergence']['iterations']
            rows.append(dict(task_id=spec['task_id'], fit_id=spec['fit_id'], original_iterations=original_iterations,
                             added_em_iterations=evidence['iterations'] - original_iterations, **evidence))
        pilot.write_csv(Path(output) / 'vmf_convergence_summary.csv', rows,
                        fields=None if rows else ['task_id', 'fit_id', 'converged', 'iterations'])
        if incomplete:
            result = dict(status='incomplete', planned_fits=len(specs), completed_fits=len(specs) - len(incomplete),
                          incomplete_tasks=incomplete, converged_vmf_fits=len(rows), input_fingerprint=summary['input_fingerprint'])
            pilot.write_json(Path(output) / 'status.json', result)
            print(json.dumps(result), flush=True)
            return 75
        # Final audit re-authenticates every original task, including copied maps.
        for spec in specs:
            original = task_path(source_results, spec['task_id'])
            identity = extension['source']['tasks'][str(spec['task_id'])]
            for name, expected in dict(identity['artifact_sha256'], **{'status.json': identity['status_sha256']}).items():
                if pilot.sha256(original / name) != expected:
                    raise ValueError('Original task artifact changed during continuation')
        from semantic_threefold_report import report_run, comparison_kind
        result = report_run(output, inputs['index'], inputs['reviews'], texts=pilot.load_texts(inputs))
        if result.get('status') != 'complete':
            raise ValueError('Reporter did not confirm complete authenticated task inventory')
        expected_comparisons = Counter()
        for left, right in combinations(specs, 2):
            kind = comparison_kind(left, right)
            if kind:
                expected_comparisons[kind] += len(set(left['eval_folds']) & set(right['eval_folds']))
        expected_comparisons = {key: count for key, count in expected_comparisons.items() if count}
        if result['comparisons'] != expected_comparisons or result['completed_fits'] != len(specs):
            raise ValueError('Revised comparison inventory differs from the unchanged design')
        result.update(converged_vmf_fits=len(rows), total_em_iteration_limit=None,
                      fit_family_counts=dict(Counter(spec['family'] for spec in specs)),
                      original_iteration_limit=config['models']['max_iter'], input_fingerprint=summary['input_fingerprint'],
                      aggregated_utc=pilot.now())
        report = Path(output) / 'RESULTS.md'
        continuation = ('## Optimization continuation\n\n'
                        f'All {len(rows)} vMF fits now satisfy the original mean log-likelihood improvement tolerance '
                        f'of {config["models"]["tolerance"]:g}. The original {config["models"]["max_iter"]}-iteration cutoff '
                        'applied to the first experiment only. Continuation resumed the saved parameters with no total EM '
                        'iteration limit; checkpoint chunks and scheduler deadlines were pause boundaries, not completion criteria. '
                        'Training posts, evaluation posts, seeds, tolerance, and concentration cap were unchanged. '
                        'Previously converged vMF fits and all other methods retain their original numerical outputs. '
                        'Continuation elapsed times include data loading and checkpoint I/O; they are not directly comparable to the original pure fitting timers. '
                        'See `vmf_convergence_summary.csv` for the exact iterations and stopping evidence for every vMF fit.\n\n')
        report.write_text(continuation + report.read_text())
        result['output_sha256']['RESULTS.md'] = pilot.sha256(report)
        result['output_sha256']['vmf_convergence_summary.csv'] = pilot.sha256(Path(output) / 'vmf_convergence_summary.csv')
        pilot.write_json(Path(output) / 'report_status.json', result)
        pilot.write_json(Path(output) / 'status.json', result)
        pilot.write_json(Path(output) / 'convergence_status.json', result)
        print(json.dumps(result), flush=True)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-results', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--task-id', type=int)
    group.add_argument('--aggregate', action='store_true')
    parser.add_argument('--wall-seconds', type=float, default=6600)
    parser.add_argument('--chunk-iterations', type=int, default=200)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            result = prepare(args.source_results, args.data_root, args.output, chunk_iterations=args.chunk_iterations,
                             checkpoint_every=args.checkpoint_every)
            print(json.dumps({'status': 'prepared', 'input_fingerprint': result['input_fingerprint'],
                              'pending_task_ids': read_json(args.output / 'extension.json')['pending_task_ids']}), flush=True)
            return 0
        if args.aggregate:
            return aggregate(args.source_results, args.data_root, args.output)
        return run_task(args.source_results, args.data_root, args.output, args.task_id, wall_seconds=args.wall_seconds)
    except (ValueError, KeyError, OSError, RuntimeError, FloatingPointError) as exc:
        print(f'Convergence extension stopped: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
