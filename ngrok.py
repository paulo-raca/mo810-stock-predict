import logging
import subprocess
import os
import requests
import time
import json

logger = logging.getLogger('ngrok')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ngrok").setLevel(logging.WARNING)

class Ngrok:
    """
    NGrok is a public service that creates tunnels from local servers, making them available over the internet.

    In this project, we'll use it to tunnel tensorboard running inside a Colaboratory server to a public address that can be accessed outside the Jupyter frontend

    FIXME: While it does work, the link seems to break after a few minutes :/
    """
    def __init__(self, *args):
        self.args = args

        if not os.path.exists("/tmp/ngrok"):
            logger.debug('Fetching ngrok')
            ret = subprocess.check_call(["wget", "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip", "-O", "/tmp/ngrok.zip"])
            if ret != 0:
                raise Exception("Failed to fetch ngrok")

            ret = subprocess.check_call(["unzip", "/tmp/ngrok.zip", "-d", "/tmp"])
            if ret != 0:
                raise Exception("Failed to unzip ngrok")

        logger.debug('Starting ngrok process')
        self.process = subprocess.Popen(["/tmp/ngrok"] + list(args) + ["--log=stderr", "--log-format=json"], stderr=subprocess.PIPE)

        try:
            self.admin_addr = None
            for line in self.process.stderr:
                line = json.loads(line)
                if line.get('obj') == 'web' and line.get('msg') == 'starting web service':
                    self.admin_addr = line['addr']
                    logger.debug(f'ngrok local address: {self.admin_addr}')
                    break

            if self.admin_addr is None:
                raise Exception("ngrok didn't bind to a local address!?")

        except:
            logger.warning('Initialization error, killing ngrok process')
            self.process.kill()
            raise

    @property
    def tunnels(self, timeout=5):
        start_time = time.time()

        while True:
            tunnels = requests.get(f'http://{self.admin_addr}/api/tunnels').json()['tunnels']
            if len(tunnels) != 0:
                return tunnels
            elif (time.time() - start_time) > timeout:
                raise TimeoutError()
            else:
                time.sleep(0.2)
                continue

    @property
    def public_url(self):
        return self.tunnels[-1]['public_url']

    def close(self):
        self.process.kill()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def __repr__(self):
        return f'Ngrok({", ".join([repr(x) for x in self.args])})'


    def __str__(self):
        return f'Ngrok.Http({self.public_url} -> ({", ".join([repr(x) for x in self.args])}))'



class Http (Ngrok):
    def __init__(self, host='localhost', port=80):
        self.host = host
        self.port = port
        Ngrok.__init__(self, 'http', f'{host}:{port}', '--bind-tls=true')

    def __repr__(self):
        return f'Ngrok.Http(host={repr(self.host)}, port={repr(self.port)})'

    def __str__(self):
        return f'Ngrok.Http({self.public_url} -> http://{self.host}:{self.port})'
