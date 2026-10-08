"""Computation-disjoint source cohorts for generated-chain probe training."""
from __future__ import annotations

import hashlib

from .generator import instance_arithmetic


def computation_key(instance):
    return instance_arithmetic(instance).key()


def generated_cohorts(source, tests, demonstrations, n_train=2000, n_validation=400):
    forbidden = {computation_key(x) for x in [*tests, *demonstrations]}
    train, validation = [], []
    ordered = sorted(source, key=lambda x: hashlib.sha256(f"round6-source:{x.id}".encode()).digest())
    for x in ordered:
        key = computation_key(x)
        if x.condition != 'neutral' or key in forbidden:
            continue
        bucket = int.from_bytes(hashlib.sha256(f"round6-generated:{key}".encode()).digest()[:8], 'big') % 5
        if bucket == 0 and len(validation) < n_validation:
            validation.append(x)
        elif bucket != 0 and len(train) < n_train:
            train.append(x)
        if len(train) == n_train and len(validation) == n_validation:
            break
    if len(train) != n_train or len(validation) != n_validation:
        raise ValueError('insufficient computation-disjoint generated-chain source rows')
    if {computation_key(x) for x in train} & {computation_key(x) for x in validation}:
        raise ValueError('generated-chain training overlaps validation computations')
    return train, validation
