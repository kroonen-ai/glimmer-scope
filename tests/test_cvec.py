import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from scope.cvec import write_control_vector


def test_control_vector_round_trip(tmp_path):
    vector = np.random.default_rng(0).standard_normal(16).astype(np.float32)
    path = tmp_path / "v.gguf"
    write_control_vector(path, 24, vector)
    reader = gguf.GGUFReader(str(path))
    (tensor,) = reader.tensors
    assert tensor.name == "direction.24"
    assert tensor.tensor_type == gguf.GGMLQuantizationType.F32
    np.testing.assert_array_equal(np.asarray(tensor.data), vector)
    arch = reader.fields["general.architecture"]
    assert bytes(arch.parts[arch.data[0]]).decode() == "controlvector"
