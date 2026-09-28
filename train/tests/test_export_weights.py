import re
import subprocess
import sys
from pathlib import Path

import net


def test_export_weights_produces_valid_c_header(tmp_path):
    params = net.init_params(seed=0)
    npz_path = tmp_path / "params.npz"
    net.save_params(params, npz_path)

    out_path = tmp_path / "model_weights.h"
    script = Path(__file__).resolve().parents[1] / "export_weights.py"
    result = subprocess.run(
        [sys.executable, str(script), str(npz_path), "--out", str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    text = out_path.read_text()
    assert "#pragma once" in text
    assert f"#define INPUT_DIM {net.INPUT_DIM}" in text
    assert f"#define H1_DIM {net.H1}" in text
    assert f"#define H2_DIM {net.H2}" in text
    assert f"#define POLICY_DIM {net.POLICY_DIM}" in text

    for cname, expected_size in [
        ("g_W1", net.INPUT_DIM * net.H1), ("g_b1", net.H1),
        ("g_W2", net.H1 * net.H2), ("g_b2", net.H2),
        ("g_Wp", net.H2 * net.POLICY_DIM), ("g_bp", net.POLICY_DIM),
        ("g_Wv", net.H2 * 1), ("g_bv", 1),
    ]:
        m = re.search(rf"static const float {cname}\[(\d+)\]", text)
        assert m, f"missing array {cname}"
        assert int(m.group(1)) == expected_size
