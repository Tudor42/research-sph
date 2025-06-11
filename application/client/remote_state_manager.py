import socket
import sys
import numpy as np
from application.utils.network import send_msg, recv_msg

class RemoteStateManager:
    """State manager communicating with a remote simulation server."""

    def __init__(self, host="127.0.0.1", port=50007, password=""):
        self.sock = socket.create_connection((host, port))
        self.dt = 0.0
        self.step = 0
        send_msg(self.sock, {"cmd": "init", "password": password})
        self._state = recv_msg(self.sock)
        if "error" in self._state:
            print("Error while connecting", self._state["error"])
            sys.exit(0)
        if self._state is not None:
            self.dt = float(self._state.get("dt", 0.0))

    def close(self):
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _update_state(self, cmd):
        send_msg(self.sock, cmd)
        self._state = recv_msg(self.sock)
        if self._state is not None and "dt" in self._state:
            self.dt = float(self._state.get("dt", 0.0))

    def get_tags(self):
        return np.array(self._state['tags'])

    def get_positions(self):
        return np.array(self._state['positions'])

    def cases_names(self):
        self._update_state({'cmd': 'cases'})
        return self._state['cases']

    def solvers_names(self):
        self._update_state({'cmd': 'solvers'})
        return self._state['solvers']

    def select_case(self, case_name):
        self._update_state({'cmd': 'select_case', 'case': case_name})
        self.step = 0

    def select_solver(self, solver_name):
        self._update_state({'cmd': 'select_solver', 'solver': solver_name})
        self.step = 0

    def reset_scene(self):
        self._update_state({'cmd': 'reset'})
        self.step = 0

    def advance(self):
        self._update_state({'cmd': 'step'})
        self.step += 1