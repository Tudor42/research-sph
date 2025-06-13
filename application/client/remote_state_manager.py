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
        self.state = recv_msg(self.sock)

        if "error" in self.state:
            print("Error while connecting", self.state["error"])
            return

        send_msg(self.sock, {'cmd': 'cases'})
        msg = recv_msg(self.sock)
        self.cases = msg["cases"]
        self.selected_case = msg["selected_case"]
        send_msg(self.sock, {'cmd': 'solvers'})
        msg = recv_msg(self.sock)
        self.solvers = msg["solvers"]
        self.selected_solver = msg["selected_solver"]

        if self.state is not None:
            self.dt = float(self.state.get("dt", 0.0))

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
        self.state = recv_msg(self.sock)
        if self.state is not None and "dt" in self.state:
            self.dt = float(self.state.get("dt", 0.0))

    def get_tags(self):
        return np.array(self.state['tags'])

    def get_positions(self):
        return np.array(self.state['positions'])

    def get_velocities(self):
        return np.array(self.state["velocities"])

    def cases_names(self):
        return self.cases

    def solvers_names(self):
        return self.solvers

    def select_case(self, case_name):
        self._update_state({'cmd': 'select_case', 'case': case_name})
        self.selected_case = case_name
        self.step = 0

    def select_solver(self, solver_name):
        self._update_state({'cmd': 'select_solver', 'solver': solver_name})
        self.selected_solver = solver_name
        self.step = 0

    def reset_scene(self):
        self._update_state({'cmd': 'reset'})
        self.step = 0

    def advance(self):
        self._update_state({'cmd': 'step'})
        self.step += 1