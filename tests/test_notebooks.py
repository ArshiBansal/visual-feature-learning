import ast
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def test_notebook_code_cells_have_valid_syntax():
    for notebook_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        with notebook_path.open(encoding="utf-8") as file:
            notebook = json.load(file)

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{notebook_path.name}:cell-{index}", "exec")


def test_notebooks_do_not_use_machine_specific_absolute_paths():
    for notebook_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        text = notebook_path.read_text(encoding="utf-8")
        assert "C:\\Users\\" not in text


def test_full_training_notebook_max_pool_backward_shape():
    notebook_path = NOTEBOOKS_DIR / "03_full_dataset_training.ipynb"

    with notebook_path.open(encoding="utf-8") as file:
        notebook = json.load(file)

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if "class ReLU:" in source and "class MaxPool2D:" in source:
            source = source[
                source.index("class ReLU:"):
                source.index("class Dense:")
            ]
            namespace = {"np": np}
            exec(compile(source, str(notebook_path), "exec"), namespace)
            break
    else:
        raise AssertionError("Could not find notebook 03 pooling classes")

    pool = namespace["MaxPool2D"](2, 2)
    x = np.ones((2, 32, 2, 2), dtype=np.float32)
    output = pool.forward(x)
    d_x = pool.backward(
        np.ones((2, 32, 1, 1), dtype=np.float32)
    )

    assert output.shape == (2, 32, 1, 1)
    assert d_x.shape == x.shape
    np.testing.assert_allclose(d_x, 0.25)


def test_full_training_notebook_preprocesses_uint8_batches():
    notebook_path = NOTEBOOKS_DIR / "03_full_dataset_training.ipynb"

    with notebook_path.open(encoding="utf-8") as file:
        notebook = json.load(file)

    preprocess_function = None
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        module = ast.parse(
            "".join(cell.get("source", []))
        )
        preprocess_function = next(
            (
                node
                for node in module.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "preprocess_batch"
            ),
            None,
        )
        if preprocess_function is not None:
            break

    if preprocess_function is None:
        raise AssertionError(
            "Could not find notebook batch preprocessor"
        )

    isolated_module = ast.Module(
        body=[preprocess_function],
        type_ignores=[],
    )
    namespace = {
        "np": np,
        "mean": np.zeros((1, 1, 1, 3), dtype=np.float32),
        "std": np.ones((1, 1, 1, 3), dtype=np.float32),
    }
    exec(
        compile(isolated_module, str(notebook_path), "exec"),
        namespace,
    )

    images = np.full(
        (2, 32, 32, 3),
        255,
        dtype=np.uint8,
    )
    batch = namespace["preprocess_batch"](images)

    assert batch.shape == (2, 3, 32, 32)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(batch, 1.0)