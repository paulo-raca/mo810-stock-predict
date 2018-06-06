from IPython.display import clear_output, Image, display, HTML
import tensorflow as tf
import subprocess
import re
import os
import logging
import ngrok
import urllib.parse

def strip_graph_consts(graph_def, max_const_size=32):
    """
    Strip large constant values from graph_def.
    """
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def = None, max_const_size=32, height=800):
    """
    Embed a visualization of the Tensorflow Graph inside the jupyter notebook.

    Code from https://stackoverflow.com/a/38192374/995480
    """

    if graph_def is None:
        graph_def = tf.get_default_graph().as_graph_def()
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()

    strip_def = strip_graph_consts(graph_def, max_const_size=max_const_size)

    code = f"""
        <script>
          function load() {{
            document.getElementById("tf-graph").pbtxt = {repr(str(strip_def))};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height: {height}px">
          <tf-graph-basic id="tf-graph"></tf-graph-basic>
        </div>
    """

    code = code.replace('"', '&quot;')
    iframe = f"""
        <iframe seamless style="width:1200px;height:{height}px;border:0" srcdoc="{code}"></iframe>
    """
    display(HTML(iframe))



class Badge:
    def __init__(self, subject, status='', color='green', url=None):
        self.url = url
        self.subject = subject
        self.status = status
        self.color = color

    def _repr_html_(self):
        def quote(x):
            return urllib.parse.quote(x).replace('-', '--')

        img = f'<img src="https://img.shields.io/badge/{quote(self.subject)}-{quote(self.status)}-{quote(self.color)}.svg">'
        if self.url is None:
            return f'{img}'
        else:
            return f'<a href="{self.url}" target="_blank">{img}</a>'



logger = logging.getLogger('tensorboard-server')
class Server:
    instances = {}

    @staticmethod
    def of(logdir="tensorboard"):
        logdir = os.path.abspath(logdir)
        if logdir not in Server.instances:
            Server.instances[logdir] = Server(logdir, closeable=False)
        return Server.instances[logdir]

    def __init__(self, logdir="tensorboard", closeable = True):
        self.closeable = closeable
        self.logdir = os.path.abspath(logdir)
        os.makedirs(self.logdir, exist_ok=True)
        logger.debug(f'Starting tensorboard process for {self.logdir}')
        self.process = subprocess.Popen(["tensorboard", "--logdir", self.logdir, "--host", '0.0.0.0', "--port", "0"], stderr=subprocess.PIPE)

        try:
            self.version = None
            self.host = None
            self.port = None
            self.ngrok = None

            for line in self.process.stderr:
                line = line.decode("utf-8").strip()
                match = re.match('TensorBoard (.*) at http://([^:]+):(\d+) .*', line)
                if match:
                    self.version = match.group(1)
                    self.host = match.group(2)
                    self.port = match.group(3)
                    break

            if self.port is None or self.version is None:
                raise Exception("tensorboard didn't bind to a local address!?")

            self.ngrok = ngrok.Http(host=self.host, port=self.port)
            self.private_url = f'http://{self.port}:{self.port}'
            self.public_url = self.ngrok.public_url

            logging.debug(f'TensorBoard running at {self.public_url}')

        except:
            logger.warning('Initialization error, killing tensorboard process')
            if self.ngrok != None:
                self.ngrok.close()
            self.process.kill()
            raise

    @property
    def runs(self):
        return os.listdir(self.logdir)

    def run_url(self, run):
        return f'{self.public_url}?run={run}'

    def clear(self):
        for run in self.runs:
            logging.debug(f'Deleting run {run}')
            subprocess.check_call(["rm", "-Rf", f"{self.logdir}/{run}"])

    def close(self):
        if self.closeable:
            self.ngrok.close()
            self.process.kill()

    def badge(self, run=None):
        if run is None:
            return Badge('tensorboard', 'all', color='yellow', url=self.public_url)
        else:
            return Badge('tensorboard', run, color='green', url=self.run_url(run))

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def __repr__(self):
        return f'Server(logdir={repr(self.logdir)})'

    def __str__(self):
        return f'Server({self.public_url} -> {repr(self.logdir)})'
