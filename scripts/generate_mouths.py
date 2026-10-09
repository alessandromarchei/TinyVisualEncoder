#!/usr/bin/env python3
"""Build sharded, Kaggle-friendly LMDB databases of AV-HuBERT mouth inputs.

Values are lossless uint8 .npy byte streams (optionally zlib compressed).
Preprocessing is intentionally completed at training time to use exactly the
same normalization as the AV-HuBERT checkpoint.
"""
import argparse
import csv
import io
import json
import os
import zlib
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm


def encode(array, compression):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    data = stream.getvalue()
    return zlib.compress(data, level=1) if compression == 'zlib' else data


def decode(blob, compression):
    if compression == 'zlib':
        blob = zlib.decompress(blob)
    return np.load(io.BytesIO(blob), allow_pickle=False)


def load_roi(filename, crop_size):
    with np.load(filename, allow_pickle=False) as obj:
        if 'data' not in obj:
            raise ValueError('missing npz key: data')
        a = obj['data']
    if a.ndim != 3:
        raise ValueError(f'expected [T,H,W], got {a.shape}')
    if a.shape[0] == 0 or a.shape[1] < crop_size or a.shape[2] < crop_size:
        raise ValueError(f'invalid dimensions for crop {crop_size}: {a.shape}')
    if a.dtype != np.uint8:
        if not np.issubdtype(a.dtype, np.number) or not np.all(np.isfinite(a)):
            raise ValueError(f'unsupported dtype/content: {a.dtype}')
        if np.min(a) < 0 or np.max(a) > 255 or not np.all(a == np.rint(a)):
            raise ValueError('non-8-bit pixel values; refusing lossy uint8 conversion')
        a = a.astype(np.uint8)
    h, w = a.shape[1:]
    top, left = (h - crop_size) // 2, (w - crop_size) // 2
    return np.ascontiguousarray(a[:, top:top + crop_size, left:left + crop_size])


def discover(datasets):
    for name, root in datasets:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'{name}: {root}')
        for path in sorted(root.rglob('*.npz')):
            relative = path.relative_to(root).with_suffix('').as_posix()
            yield name, path, f'{name}/{relative}'


def make_env(path, map_size_gb):
    return lmdb.open(str(path), subdir=False, map_size=int(map_size_gb * 1024**3),
                     max_dbs=1, lock=True, writemap=False, map_async=False,
                     readahead=False, meminit=False)


def build(args):
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob('inputs_*.lmdb')) or (out / 'manifest.tsv').exists():
        raise RuntimeError(f'Output directory already contains LMDB/manifest: {out}. '
                           'Use a new empty directory to avoid overwriting data.')
    datasets = [('lrs2', args.lrs2), ('lrs3', args.lrs3), ('voxceleb2', args.voxceleb2)]
    if len({Path(p).resolve() for _, p in datasets}) != 3:
        raise ValueError('Each dataset must have a distinct root directory')
    max_bytes = int(args.shard_size_gb * 1024**3)
    if args.map_size_gb <= args.shard_size_gb * 1.2:
        raise ValueError('--map-size-gb must be > 1.2 * --shard-size-gb')
    summary = dict(format='numpy-npy-uint8', compression=args.compression,
                   crop_size=args.crop_size, frame_layout='T,H,W',
                   preprocessing='center-crop only; normalize during training',
                   dataset_roots={name: str(Path(root).resolve()) for name, root in datasets})
    seen = set()
    shard_id = 0
    bytes_in_shard = 0
    rows_in_shard = 0
    total, failed = 0, 0
    env = None
    txn = None
    manifest_path = out / 'manifest.tsv'
    errors_path = out / 'errors.tsv'

    def new_shard():
        nonlocal env, txn, shard_id, bytes_in_shard, rows_in_shard
        if txn is not None:
            txn.commit()
            env.sync()
            env.close()
        shard_name = f'inputs_{shard_id:04d}.lmdb'
        env = make_env(out / shard_name, args.map_size_gb)
        txn = env.begin(write=True)
        bytes_in_shard, rows_in_shard = 0, 0
        shard_id += 1
        return shard_name

    shard_name = new_shard()
    with open(manifest_path, 'w', newline='') as mf, open(errors_path, 'w', newline='') as ef:
        mw = csv.writer(mf, delimiter='\t')
        ew = csv.writer(ef, delimiter='\t')
        mw.writerow(['dataset', 'key', 'shard', 'frames', 'height', 'width', 'source_relative'])
        ew.writerow(['dataset', 'source', 'error'])
        for dataset, filename, key in tqdm(discover(datasets), desc='Mouth ROI -> LMDB'):
            if key in seen:
                raise RuntimeError(f'Duplicate key: {key}')
            seen.add(key)
            try:
                roi = load_roi(filename, args.crop_size)
                payload = encode(roi, args.compression)
            except Exception as exc:
                failed += 1
                ew.writerow([dataset, str(filename), str(exc)])
                continue
            # Keys are UTF-8 names; values are NPY byte streams.
            additional = len(key.encode()) + len(payload) + 128
            if rows_in_shard and bytes_in_shard + additional > max_bytes:
                shard_name = new_shard()
            try:
                inserted = txn.put(key.encode('utf-8'), payload, overwrite=False)
                if not inserted:
                    raise RuntimeError(f'Duplicate LMDB key: {key}')
            except lmdb.MapFullError as exc:
                raise RuntimeError('LMDB map full. Increase --map-size-gb and rebuild.') from exc
            rows_in_shard += 1
            total += 1
            bytes_in_shard += additional
            relative = key.split('/', 1)[1] + '.npz'
            mw.writerow([dataset, key, shard_name, *roi.shape, relative])
            if rows_in_shard % args.commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
            if total % 1000 == 0:
                mf.flush()
                ef.flush()
    txn.commit()
    env.sync()
    env.close()
    summary.update(samples=total, failures=failed, shards=shard_id)
    (out / 'metadata.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Manifest: {manifest_path}; Errors: {errors_path}')


class AVHubertInputLMDB:
    """Lazy read-only loader, safe with PyTorch DataLoader worker processes.

    Returns uint8 [T,H,W]. Apply the teacher's exact transforms separately.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / 'metadata.json').read_text())
        with open(self.directory / 'manifest.tsv', newline='') as f:
            self.rows = list(csv.DictReader(f, delimiter='\t'))
        self.envs = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        shard = row['shard']
        if shard not in self.envs:
            self.envs[shard] = lmdb.open(str(self.directory / shard), subdir=False,
                                        readonly=True, lock=False, readahead=False,
                                        max_readers=256)
        with self.envs[shard].begin(write=False) as txn:
            payload = txn.get(row['key'].encode('utf-8'))
        if payload is None:
            raise KeyError(row['key'])
        return row['key'], decode(payload, self.meta['compression'])

    def close(self):
        for env in self.envs.values():
            env.close()
        self.envs.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['envs'] = {}
        return state


def inspect(args):
    data = AVHubertInputLMDB(args.output_dir)
    print('Metadata:', json.dumps(data.meta, indent=2))
    print('Samples:', len(data))
    for idx in range(min(3, len(data))):
        key, arr = data[idx]
        print(f'{key}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}')
    data.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['build', 'inspect'], default='build')
    p.add_argument('--lrs2', help='LRS2 mouths directory')
    p.add_argument('--lrs3', help='LRS3 mouths directory')
    p.add_argument('--voxceleb2', help='VoxCeleb2 mouths directory')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--crop-size', type=int, default=88)
    p.add_argument('--compression', choices=['none', 'zlib'], default='zlib')
    p.add_argument('--shard-size-gb', type=float, default=2.0)
    p.add_argument('--map-size-gb', type=float, default=4.0)
    p.add_argument('--commit-every', type=int, default=128)
    args = p.parse_args()
    if args.mode == 'build':
        if not all([args.lrs2, args.lrs3, args.voxceleb2]):
            p.error('--lrs2, --lrs3 and --voxceleb2 are required for build')
        if args.crop_size <= 0 or args.commit_every <= 0 or args.shard_size_gb <= 0:
            p.error('Sizes and commit interval must be positive')
        build(args)
    else:
        inspect(args)


if __name__ == '__main__':
    main()
