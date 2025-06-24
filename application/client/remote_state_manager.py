import socket
import numpy as np
from application.state_manager import StateManager
from application.utils.network import send_msg, recv_msg

class RemoteStateManager(StateManager):
    """State manager communicating with a remote simulation server."""

    def __init__(self, host="127.0.0.1", port=50007, password=""):
        self.sock = socket.create_connection((host, port))
        self._send_msg({"cmd": "init", "password": password})
 
        self.timestamp = 0.0
        self.save_folder = "/tmp"
        self.cases = []
        self.solvers = []
        self.selected_solver = ""
        self.selected_case = ""
        self._send_msg({'cmd': 'cases'})
        self._send_msg({'cmd': 'solvers'})


    def _send_msg(self, msg):
        send_msg(self.sock, msg)
        msg = recv_msg(self.sock)
        if "error" in msg:
            raise RuntimeError(msg["error"])
        if "save_folder" in msg:
            self.save_folder = msg["save_folder"]
        if "state" in msg:
            self.state = msg["state"]
            # save self.state in save_folder, the frames will be saved in save_folder under frames subfolder
        if "curr_timestamp" in msg:
            self.timestamp = msg["curr_timestamp"]
        if "cases" in msg:
            self.cases = msg["cases"]
        if "solvers" in msg:
            self.solvers = msg["solvers"]
        if "selected_solver" in msg:
            self.selected_solver = msg["selected_solver"]
        if "selected_case" in msg:
            self.selected_case = msg["selected_case"]

    def get_tags(self):
        if self.state is not None:
            return np.array(self.state['tag'])
        else:
            return np.array()

    def get_positions(self):
        if self.state is not None:
            return np.array(self.state['positions'])
        else:
            return np.array()

    def get_velocities(self):
        if self.state is not None:
            return np.array(self.state['velocities'])
        else:
            return np.array()

    def cases_names(self):
        return self.cases

    def solvers_names(self):
        return self.solvers

    def select_case(self, case_name):
        self._send_msg({'cmd': 'select_case', 'case': case_name})

    def select_solver(self, solver_name):
        self._send_msg({'cmd': 'select_solver', 'solver': solver_name})

    def reset_scene(self):
        self._send_msg({'cmd': 'reset'})

    def advance(self):
        self._send_msg({'cmd': 'step'})

    def get_current_save_directory(self) -> str:
        return self.save_folder

    def get_timestamp(self) -> float:
        return self.timestamp

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
