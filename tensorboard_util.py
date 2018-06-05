from IPython.display import clear_output, Image, display, HTML
import tensorflow as tf
import numpy as np

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
