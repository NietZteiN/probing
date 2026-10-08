from dataclasses import replace

import pytest

from cueconf.generator import sample_single
from cueconf.round6 import computation_key, generated_cohorts


def test_generated_cohorts_keep_renamed_equations_together_and_exclude_test():
    source = list(sample_single(3, 1000, 61003, 'neutral'))
    test = replace(source[0], names={'v1':'new_name','v2':'another_name'})
    demonstration = source[1]
    train, validation = generated_cohorts(source,[test],[demonstration],n_train=100,n_validation=50)
    train_keys = {computation_key(x) for x in train}
    validation_keys = {computation_key(x) for x in validation}
    assert not train_keys & validation_keys
    assert computation_key(test) not in train_keys | validation_keys
    assert computation_key(demonstration) not in train_keys | validation_keys


def test_generated_cohorts_reject_insufficient_disjoint_source_pool():
    source = list(sample_single(3, 5, 61003, 'neutral'))
    with pytest.raises(ValueError,match='insufficient'):
        generated_cohorts(source,source,[],n_train=1,n_validation=1)
