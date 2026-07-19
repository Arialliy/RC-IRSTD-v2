from __future__ import annotations

import pytest
import torch

from data_ext.balanced_domain_loader import (
    BalancedDomainLoader as CanonicalBalancedDomainLoader,
)
from data_ext.balanced_domain_loader import (
    merge_domain_batches as canonical_merge_domain_batches,
)
from data_ext.eval_dataset import IRSTDEvalDataset as CanonicalEvalDataset
from data_ext.split_utils import (
    ensure_unique_sample_ids as canonical_ensure_unique_sample_ids,
)
from data_ext.split_utils import read_split_file as canonical_read_split_file
from data_ext.split_utils import resolve_split_file as canonical_resolve_split_file
from data_ext.split_utils import sample_id_from_entry as canonical_sample_id_from_entry
from model.MSHNet import MSHNet as CanonicalMSHNet
from rc_irstd.data.balanced import BalancedDomainLoader, merge_domain_batches
from rc_irstd.data.dataset import (
    IRSTD_Dataset,
    IRSTDEvalDataset,
    ensure_unique_sample_ids,
    read_split_file,
    resolve_split_file,
    sample_id_from_entry,
)
from rc_irstd.models.mshnet import (
    MSHNet,
    build_mshnet,
    forward_mshnet,
    structure_mshnet_output,
)
from utils.data import IRSTD_Dataset as CanonicalTrainingDataset


def test_data_facade_exports_canonical_implementations() -> None:
    assert IRSTD_Dataset is CanonicalTrainingDataset
    assert IRSTDEvalDataset is CanonicalEvalDataset
    assert BalancedDomainLoader is CanonicalBalancedDomainLoader
    assert merge_domain_batches is canonical_merge_domain_batches
    assert read_split_file is canonical_read_split_file
    assert resolve_split_file is canonical_resolve_split_file
    assert sample_id_from_entry is canonical_sample_id_from_entry
    assert ensure_unique_sample_ids is canonical_ensure_unique_sample_ids


@pytest.mark.parametrize("warm_flag", [False, True])
def test_mshnet_facade_preserves_forward_and_checkpoint_contract(
    warm_flag: bool,
) -> None:
    torch.manual_seed(7)
    canonical = CanonicalMSHNet(3).eval()
    facade = build_mshnet(3).eval()

    assert MSHNet is CanonicalMSHNet
    assert type(facade) is CanonicalMSHNet
    incompatible = facade.load_state_dict(canonical.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert tuple(facade.state_dict()) == tuple(canonical.state_dict())

    inputs = torch.randn(1, 3, 16, 16)
    with torch.inference_mode():
        expected_auxiliary, expected_logits = canonical(inputs, warm_flag)
        raw_facade = facade(inputs, warm_flag)
        structured = structure_mshnet_output(raw_facade)
        forwarded = forward_mshnet(facade, inputs, warm_flag=warm_flag)

    assert structured.logits is raw_facade[1]
    assert len(structured.auxiliary_logits) == len(expected_auxiliary)
    assert torch.equal(structured.logits, expected_logits)
    assert torch.equal(forwarded.logits, expected_logits)
    for actual, expected in zip(structured.auxiliary_logits, expected_auxiliary):
        assert torch.equal(actual, expected)
    for actual, expected in zip(forwarded.auxiliary_logits, expected_auxiliary):
        assert torch.equal(actual, expected)


def test_structured_output_rejects_noncanonical_shape() -> None:
    with pytest.raises(TypeError, match="two-item"):
        structure_mshnet_output(torch.zeros(1))
