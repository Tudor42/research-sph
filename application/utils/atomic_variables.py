import threading

class AtomicBoolean:
    def __init__(self, initial: bool = False):
        self._value = initial
        self._lock = threading.Lock()

    def get(self) -> bool:
        with self._lock:
            return self._value

    def set(self, new_value: bool):
        with self._lock:
            self._value = new_value

    def compare_and_set(self, expect: bool, update: bool) -> bool:
        with self._lock:
            if self._value == expect:
                self._value = update
                return True
            return False
