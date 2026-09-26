import nbformat
from IPython.core.interactiveshell import InteractiveShell
import io


def extract_result(path, variable_name):
    path = '/home/jovyan' + path.replace('~', '')
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    shell = InteractiveShell()

    try:
        with io.StringIO() as output_buffer:
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    shell.run_cell(cell.source)
        captured_variable = shell.user_ns.get(variable_name)
    finally:
        # restore the variable / save on memory
        del shell

    return captured_variable

def extract_results(path, *variable_names):
    captured_variables = []
    path = '/home/jovyan' + path.replace('~', '')
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    shell = InteractiveShell()

    try:
        with io.StringIO() as output_buffer:
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    shell.run_cell(cell.source)
        for variable_name in variable_names:
            captured_variables.append(shell.user_ns.get(variable_name))
            
    finally:
        # restore the variable / save on memory
        del shell

    return captured_variables