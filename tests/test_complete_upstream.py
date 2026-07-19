from __future__ import annotations

from collections import OrderedDict

import torch

from rc_irstd.models import MSHNet
from rc_irstd.utils.upstream import convert_upstream_mshnet_state


def test_public_mshnet_name_conversion_preserves_canonical_keys() -> None:
    model = MSHNet(3)
    upstream = OrderedDict(
        ("module." + key, value.detach().clone())
        for key, value in model.state_dict().items()
    )
    converted = convert_upstream_mshnet_state(upstream)
    incompatible = model.load_state_dict(converted, strict=False)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert list(converted) == list(model.state_dict())
    for key, expected in model.state_dict().items():
        assert torch.equal(converted[key], expected)
