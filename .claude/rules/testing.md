---
paths:
  - "tests/**/*.py"
---

# Testing Conventions

## Runner
- Use `uv run python -m pytest` to run tests
- Use `pytest.mark.parametrize` for data-driven tests
- Group related tests in classes when they share setup

## Fixtures
- Define shared fixtures in `tests/conftest.py`
- Use `tmp_path` for temporary file/directory needs

## Mocking Heavy Dependencies
- Always mock `torch`, `transformers`, and `sentence-transformers` — never load real models in tests
- Use `unittest.mock.patch` to stub model loading and inference
- Mock `cv2.VideoCapture` for video-dependent tests — provide fake frame arrays via numpy

## Assertions
- Use plain `assert` statements (not `self.assertEqual`)
- Compare numpy arrays with `np.testing.assert_array_equal` or `np.allclose`
- Check approximate floats with `pytest.approx`
