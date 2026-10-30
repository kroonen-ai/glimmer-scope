import torch

from scope.crosslayer import CoFire


def dense(vals, idx, n):
    out = torch.zeros(vals.shape[0], n)
    out.scatter_add_(1, idx, vals)
    return out


def encodings(tokens, n, k, gen):
    vals = torch.rand(tokens, k, generator=gen)
    vals[vals < 0.2] = 0
    idx = torch.stack([torch.randperm(n, generator=gen)[:k] for _ in range(tokens)])
    return vals, idx


def test_partners_match_dense_cosine():
    gen = torch.Generator().manual_seed(0)
    n, k, top = 40, 5, 3
    acc = CoFire(n, 3, device="cpu", block_rows=16)
    batches = [[encodings(30, n, k, gen) for _ in range(3)] for _ in range(4)]
    for enc in batches:
        acc.add(enc)
    assert acc.tokens == 120
    a = [torch.cat([dense(*b[layer], n) for b in batches]) for layer in range(3)]
    found = acc.partners(top, chunk=7)
    for pair in range(2):
        lo, hi = a[pair], a[pair + 1]
        cos = (lo.t() @ hi) / (lo.norm(dim=0)[:, None] * hi.norm(dim=0)[None, :]).clamp_min(1e-12)
        up_cos, _ = cos.topk(top, dim=1)
        down_cos, _ = cos.t().topk(top, dim=1)
        assert torch.allclose(torch.from_numpy(found[pair]["up_cos"]), up_cos, atol=1e-5)
        assert torch.allclose(torch.from_numpy(found[pair]["down_cos"]), down_cos, atol=1e-5)
        got = cos.gather(1, torch.from_numpy(found[pair]["up_idx"]).long())
        assert torch.allclose(got, up_cos, atol=1e-5)
        got_down = cos.t().gather(1, torch.from_numpy(found[pair]["down_idx"]).long())
        assert torch.allclose(got_down, down_cos, atol=1e-5)
    assert torch.equal(acc.freq[0], (a[0] > 0).sum(0))
