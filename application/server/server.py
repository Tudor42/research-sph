import socket
import threading
import numpy as np
from application.server.managers.state_manager import StateManager
from application.utils.network import recv_msg, send_msg

_connection_count = 0
_conn_count_lock = threading.Lock()

def make_payload(state_manager):
    return {
        'tags': np.array(state_manager.get_tags()),
        'positions': np.array(state_manager.get_positions()),
        'velocities': np.array(state_manager.get_velocities()),
        'dt': state_manager.dt,
    }

def _process_command(sm, cmd):
    name = cmd.get("cmd")
    if name == "init":
        return make_payload(sm)
    if name == "step":
        sm.advance()
        return make_payload(sm)
    if name == "reset":
        sm.reset_scene()
        return make_payload(sm)
    if name == "select_case":
        sm.select_case(cmd.get("case"))
        return make_payload(sm)
    if name == "select_solver":
        sm.select_solver(cmd.get("solver"))
        return make_payload(sm)
    if name == "cases":
        return {"cases": sm.cases_names(), "selected_case": sm.case_manager.curr_case_name}
    if name == "solvers":
        return {"solvers": sm.solvers_names(), "selected_solver": sm.solver_manager.curr_solver_name}
    return None


def handle_client(sm, conn: socket.socket, addr, password):
    print(f"{addr} connected")
    global _connection_count
    with _conn_count_lock:
        if _connection_count >= 1:
            send_msg(conn, {"error": "Server busy: only one client allowed"})
            conn.close()
            return
        _connection_count += 1
    conn.settimeout(3.0)
    try:
        cmd = recv_msg(conn)
        if not cmd or cmd.get("cmd") != "init":
            send_msg(conn, {"error": "Authentification failed: invalid initial message"})
            return
        if cmd.get("password") != password:
            send_msg(conn, {"error": "Authentification failed: invalid password"})
            return
        send_msg(conn, make_payload(sm))
        conn.settimeout(None)
        while True:
            try:
                cmd = recv_msg(conn)
                if cmd is None:
                    break
            except (ConnectionResetError, BrokenPipeError):
                print(f"Broken pipe from {addr}")
                break
            except socket.timeout:
                print(f"Timeout waiting for init from {addr}")
                break
            try:
                result = _process_command(sm, cmd)
            except Exception as e:
                err_msg = str(e)
                print(f"Error processing command from {addr}: {err_msg}")
                try:
                    send_msg(conn, {"error": err_msg})
                except Exception:
                    pass
                break
            if result is not None:
                send_msg(conn, result)
    except socket.timeout:
        print(f"Timeout waiting for init from {addr}")
    finally:
        print(f"{addr} disconnected")
        with _conn_count_lock:
            _connection_count -= 1
        conn.close()

def main(password, host="127.0.0.1", port=50007):
    sm = StateManager()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        while True:
            conn, addr = s.accept()

            thread = threading.Thread(
                target=handle_client,
                args=(sm, conn, addr, password),
                daemon=True
            )
            thread.start()
