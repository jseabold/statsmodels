#! /usr/bin/env python
"""
Script to generate notebooks with output from notebooks that don't have
output.
"""
from __future__ import print_function

# prefer HTML over rST for now until nbconvert changes drop
OUTPUT = "html"

import os
import io
import sys
import json
import logging
import platform
from time import sleep

SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "examples",
                                          "notebooks"))

try:
    from Queue import Empty
except ImportError:  # Python 3
    from queue import Empty

from IPython.kernel import KernelManager

from IPython.nbformat import reads, NotebookNode, convert

cur_dir = os.path.abspath(os.path.dirname(__file__))

# for conversion of .ipynb -> html/rst

from IPython.config import Config
try:
    from IPython.nbconvert.exporters import HTMLExporter
except ImportError:
    from warnings import warn
    from statsmodels.tools.sm_exceptions import ModuleUnavailableWarning
    warn("Notebook examples not built. You need IPython >= 3.x.",
         ModuleUnavailableWarning)
    sys.exit(0)

import hash_funcs


class NotebookError(Exception):
    pass


class NotebookRunner(object):
    # The kernel communicates with mime-types while the notebook
    # uses short labels for different cell types. We'll use this to
    # map from kernel types to notebook format types.

    MIME_MAP = {
        'image/jpeg': 'jpeg',
        'image/png': 'png',
        'text/plain': 'text',
        'text/html': 'html',
        'text/latex': 'latex',
        'application/javascript': 'html',
        'image/svg+xml': 'svg',
    }

    def __init__(self, nb, mpl_inline=False, profile_dir=None,
                 working_dir=None):
        self.km = KernelManager()

        args = []

        if profile_dir:
            args.append('--profile-dir=%s' % os.path.abspath(profile_dir))

        cwd = os.getcwd()

        if working_dir:
            os.chdir(working_dir)

        self.km.start_kernel(extra_arguments=args)

        os.chdir(cwd)

        if platform.system() == 'Darwin':
            # There is sometimes a race condition where the first
            # execute command hits the kernel before it's ready.
            # It appears to happen only on Darwin (Mac OS) and an
            # easy (but clumsy) way to mitigate it is to sleep
            # for a second.
            sleep(1)

        self.kc = self.km.client()
        self.kc.start_channels()
        try:
            self.kc.wait_for_ready()
        except AttributeError:
            # IPython < 3
            self._wait_for_ready_backport()

        if mpl_inline:
            self.kc.execute("%matplotlib inline")

        self.nb = nb

    def shutdown_kernel(self):
        logging.info('Shutdown kernel')
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)

    def _wait_for_ready_backport(self):
        """Backport BlockingKernelClient.wait_for_ready from IPython 3"""
        # Wait for kernel info reply on shell channel
        self.kc.kernel_info()
        while True:
            msg = self.kc.get_shell_msg(block=True, timeout=30)
            if msg['msg_type'] == 'kernel_info_reply':
                break

        # Flush IOPub channel
        while True:
            try:
                msg = self.kc.get_iopub_msg(block=True, timeout=0.2)
            except Empty:
                break

    def run_cell(self, cell):
        '''
        Run a notebook cell and update the output of that cell in-place.
        '''
        logging.info('Running cell:\n%s\n', cell.source)
        self.kc.execute(cell.source)
        reply = self.kc.get_shell_msg()
        status = reply['content']['status']
        if status == 'error':
            traceback_text = 'Cell raised uncaught exception: \n' + \
                '\n'.join(reply['content']['traceback'])
            logging.info(traceback_text)
        else:
            logging.info('Cell returned')

        outs = list()
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=90)
                if msg['msg_type'] == 'status':
                    if msg['content']['execution_state'] == 'idle':
                        break
            except Empty:
                # execution state should return to idle before the queue
                # becomes empty, if it doesn't, something bad has happened
                raise

            content = msg['content']
            msg_type = msg['msg_type']

            # IPython 3.0.0-dev writes pyerr/pyout in the notebook format but
            # uses error/execute_result in the message spec. This does the
            # translation needed for tests to pass with IPython 3.0.0-dev
            notebook3_format_conversions = {
                'error': 'pyerr',
                'execute_result': 'pyout'
            }
            msg_type = notebook3_format_conversions.get(msg_type, msg_type)

            out = NotebookNode(output_type=msg_type)

            if 'execution_count' in content:
                cell['prompt_number'] = content['execution_count']
                out.prompt_number = content['execution_count']

            if msg_type in ('status', 'pyin', 'execute_input'):
                continue
            elif msg_type == 'stream':
                out.stream = content['name']
                out.name = content['name']  # for v4
                # in msgspec 5, this is name, text
                # in msgspec 4, this is name, data
                if 'text' in content:
                    out.text = content['text']
                else:
                    out.text = content['data']
                # print(out.text, end='')
            elif msg_type in ('display_data', 'pyout'):
                for mime, data in content['data'].items():
                    try:
                        attr = self.MIME_MAP[mime]
                    except KeyError:
                        raise NotImplementedError('unhandled mime type: %s'
                                                  % mime)

                    setattr(out, attr, data)
                # print(data, end='')
            elif msg_type == 'pyerr':
                out.ename = content['ename']
                out.evalue = content['evalue']
                out.traceback = content['traceback']

                # logging.error('\n'.join(content['traceback']))
            elif msg_type == 'clear_output':
                outs = list()
                continue
            else:
                raise NotImplementedError('unhandled iopub message: %s' %
                                          msg_type)
            outs.append(out)
        cell['outputs'] = outs

        if status == 'error':
            raise NotebookError(traceback_text)

    def iter_code_cells(self):
        '''
        Iterate over the notebook cells containing code.
        '''
        for cell in self.nb.cells:
            if cell.cell_type == 'code':
                yield cell

    def run_notebook(self, skip_exceptions=False, progress_callback=None):
        '''
        Run all the cells of a notebook in order and update
        the outputs in-place.

        If ``skip_exceptions`` is set, then if exceptions occur in a cell, the
        subsequent cells are run (by default, the notebook execution stops).
        '''
        for i, cell in enumerate(self.iter_code_cells()):
            try:
                self.run_cell(cell)
            except NotebookError:
                if not skip_exceptions:
                    raise
            if progress_callback:
                progress_callback(i)

    def count_code_cells(self):
        '''
        Return the number of code cells in the notebook
        '''
        return sum(1 for _ in self.iter_code_cells())


def _get_parser():
    try:
        import argparse
    except ImportError:
        raise ImportError("This script only runs on Python >= 2.7")
    parser = argparse.ArgumentParser(description="Convert .ipynb notebook "
                                                 "inputs to HTML page with "
                                                 "output")
    parser.add_argument("path", type=str, default=SOURCE_DIR, nargs="?",
                        help="path to folder containing notebooks")
    parser.add_argument("--profile", type=str,
                        help="profile name to use")
    parser.add_argument("--timeout", default=90, type=int,
                        metavar="N",
                        help="how long to wait for cells to run in seconds")
    return parser


def nb2html(nb):
    """
    Cribbed from nbviewer
    """
    config = Config()
    config.HTMLExporter.template_file = 'basic'
    config.NbconvertApp.fileext = "html"
    config.CSSHtmlHeaderTransformer.enabled = False

    C = HTMLExporter(config=config)
    return C.from_notebook_node(nb)[0]


if __name__ == '__main__':
    rst_target_dir = os.path.join(cur_dir, '..',
                                  'docs/source/examples/notebooks/generated/')
    if not os.path.exists(rst_target_dir):
        os.makedirs(rst_target_dir)

    is_nb = lambda x : x.endswith('.ipynb') and 'generated' not in x
    contents = os.listdir(SOURCE_DIR)
    notebooks = [os.path.join(SOURCE_DIR, i) for i in filter(is_nb, contents)]

    try:
        for fname in notebooks:
            f = open(fname, 'r').read()
            base, ext = os.path.splitext(fname)
            fname_only = os.path.basename(base)
            # check if we need to write
            towrite, filehash = hash_funcs.check_hash(f, fname_only)
            if not towrite:
                print("Hash has not changed for file %s" % fname_only)
                continue
            print("Writing ", fname_only)
            nb = json.loads(f)
            nb_version = nb['nbformat']
            nb = reads(f, as_version=nb_version)
            if nb_version != 4:
                nb = convert(nb, 4)
            notebook_runner = NotebookRunner(nb, working_dir=SOURCE_DIR,
                                             profile_dir=None,
                                             mpl_inline=True)

            # This edits the notebook cells inplace
            notebook_runner.run_notebook(skip_exceptions=True)

            # use nbconvert to convert to rst
            support_file_dir = os.path.join(rst_target_dir,
                                            fname_only+"_files")
            if OUTPUT == "html":
                from notebook_output_template import notebook_template
                new_html = os.path.join(rst_target_dir, fname_only+".rst")
                # get the title out of the notebook because sphinx needs it
                title_cell = nb['cells'].pop(0)
                if title_cell['cell_type'] == 'heading':
                    pass
                elif (title_cell['cell_type'] == 'markdown'
                      and title_cell['source'].strip().startswith('#')):
                    # IPython 3.x got rid of header cells
                    pass
                else:
                    print("Title not in first cell for ", fname_only)
                    print("Not generating rST")
                    continue

                html_out = nb2html(nb)
                # indent for insertion into raw html block in rST
                html_out = "\n".join(["   "+i for i in html_out.split("\n")])
                with io.open(new_html, "w", encoding="utf-8") as f:
                    f.write(title_cell["source"].replace("#",
                                                         "").strip() + u"\n")
                    f.write(u"="*len(title_cell["source"])+u"\n\n")
                    f.write(notebook_template.substitute(name=fname_only,
                                                         body=html_out))
            hash_funcs.update_hash_dict(filehash, fname_only)
    except Exception as err:
        raise err

    finally:
        os.chdir(cur_dir)
