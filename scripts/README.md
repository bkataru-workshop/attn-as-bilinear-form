# LaTeX Linter for Markdown Files

This script checks for common LaTeX formatting errors in markdown files, particularly in inline math (`$...$`) and display math (`$$...$$`) blocks.

## Usage

### Direct usage

```bash
python3 scripts/lint_latex.py <file1.md> <file2.md> ...
```

### Using justfile

```bash
# Run only LaTeX linter
just lint-latex

# Run all linters (Vale + LaTeX)
just lint-all
```

## What it checks

The linter checks for subscripts and superscripts with multi-letter words that should be wrapped in `\text{}`:

### ❌ Incorrect

```latex
$X_{input}$        # Should be $X_{\text{input}}$
$W_{QKV}$          # Should be $W_{\text{QKV}}$
$D_{KL}$           # Should be $D_{\text{KL}}$
```

### ✅ Correct

```latex
$X_{\text{input}}$ # Multi-letter word wrapped in \text{}
$Q^{ia}$           # Single letters (tensor notation)
$Q^{hia}$          # Multiple single letters (tensor notation)
$d_{model}$        # Common dimension name (exception)
$x_{max}$          # Math operator (exception)
```

## Exceptions

The linter allows certain patterns without requiring `\text{}`:

1. **Single letters**: `_i`, `_j`, `^a`, `^b`
2. **Tensor indices**: `_{ab}`, `^{ij}`, `_{hia}` (all lowercase letters)
3. **Math operators**: `min`, `max`, `sin`, `cos`, `log`, etc.
4. **Dimension names**: `model` (as in `d_model`)

## Why this matters

LaTeX subscripts and superscripts with multi-letter words can render incorrectly in MathJax/KaTeX if not wrapped in `\text{}`. For example:

- `$X_{input}$` renders as: X<sub>i</sub><sub>n</sub><sub>p</sub><sub>u</sub><sub>t</sub> (italicized, spaced incorrectly)
- `$X_{\text{input}}$` renders as: X<sub>input</sub> (correct, upright text)

## Adding to CI

To add this linter to your CI pipeline:

```yaml
- name: Lint LaTeX
  run: python3 scripts/lint_latex.py $(find . -name "*.md" -type f -not -path "./.git/*")
```

## Extending the linter

To add more patterns or exceptions, edit `scripts/lint_latex.py`:

- Add patterns to `self.patterns` in the `__init__` method
- Add exceptions to `_is_valid_subscript()` or `_is_valid_superscript()`
