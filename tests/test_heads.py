import torch

from wev.model import PointerHead, SetHead


def test_set_head_starts_equal_to_pointer_head():
    torch.manual_seed(0)
    head = SetHead(64, dp=32)
    ref = PointerHead(64, dp=32)
    ref.load_state_dict(head.pointer.state_dict())
    hd, ho = torch.randn(64), torch.randn(7, 64)
    assert torch.allclose(head(hd, ho), ref(hd, ho), atol=1e-6)


def test_set_head_is_permutation_equivariant():
    torch.manual_seed(0)
    head = SetHead(64, dp=32)
    for p in head.out.parameters():   # make the refinement non-trivial
        torch.nn.init.normal_(p)
    hd, ho = torch.randn(64), torch.randn(9, 64)
    perm = torch.randperm(9)
    assert torch.allclose(head(hd, ho)[perm], head(hd, ho[perm]), atol=1e-5)


def test_set_head_handles_a_single_option():
    assert SetHead(64, dp=32)(torch.randn(64), torch.randn(1, 64)).shape == (1,)
