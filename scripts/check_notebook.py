"""Validate or execute the notebook's plain-Python cells in one IPython session.

This in-process check needs no separate Jupyter kernel or connection files.
IPython transforms shell commands and magics before syntax validation.
Generated outputs are saved only when --save-outputs is explicitly requested.
"""
from pathlib import Path
import os
import argparse
import nbformat
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true', help='Check notebook structure and Python syntax without data or outputs.')
    parser.add_argument('--save-outputs', action='store_true', help='Save generated cell outputs locally; clear them before committing.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / 'notebooks/modelling.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if args.validate_only:
        cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        shell = InteractiveShell.instance()
        for count, cell in enumerate(cells, 1):
            compile(shell.input_transformer_manager.transform_cell(cell.source), f'{path.name}:cell-{count}', 'exec')
        print(f'Validated notebook structure and syntax of {len(cells)} code cells without data.')
        return
    os.chdir(path.parent)
    shell = InteractiveShell.instance()
    configure_inline_support(shell, 'module://matplotlib_inline.backend_inline')
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=False)
        result.raise_error()
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
    nbformat.validate(notebook)
    if args.save_outputs:
        nbformat.write(notebook, path)
    print(f'Executed and validated {count} code cells without errors (in-process IPython).')

if __name__ == '__main__':
    main()
