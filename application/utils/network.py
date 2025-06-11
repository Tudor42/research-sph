"""Helpers for sending and receiving pickled messages over sockets."""

import pickle
import struct

# Helper functions for sending and receiving Python objects over sockets

_HEADER_STRUCT = struct.Struct('!I')  # Network order unsigned int


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER_STRUCT.pack(len(data)) + data)


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        packet = sock.recv(n - len(buf))
        if not packet:
            return None
        buf.extend(packet)
    return bytes(buf)


def recv_msg(sock):
    header = _recvall(sock, _HEADER_STRUCT.size)
    if not header:
        return None
    (size,) = _HEADER_STRUCT.unpack(header)
    data = _recvall(sock, size)
    if data is None:
        return None
    return pickle.loads(data)