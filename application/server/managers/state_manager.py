from application.server.managers.case_manager import CaseManager
from application.server.managers.solver_manager import SolverManager
import jax
import jax.numpy as jnp
import time
from application.state_manager import StateManager

class StateManagerImpl(StateManager):
    def __init__(self):
        self.case_manager = CaseManager()
        self.solver_manager = SolverManager()
        
        self.case_manager.select("db")
        self.solver_manager.select("wcsph")
        self.dt = self.case_manager.cfg.solver.dt
        self.step = 0
        self.curr_timestamp = 0.0

        self.set_save_directory()
        self.reset_scene()

    def set_dt(self):
        self.dt = self.case_manager.cfg.solver.dt
        if self.solver_manager.curr_solver_name != "wcsph":
            self.dt = 100 * self.dt

    def set_save_directory(self):
        self.curr_save_directory = self.case_manager.curr_case_name + "_" + self.solver_manager.curr_solver_name + "_" + str(time.time())

    def cases_names(self):
        return self.case_manager.list_names()

    def solvers_names(self):
        return self.solver_manager.list_names()

    def get_tags(self):
        return self.state["tag"]

    def get_positions(self):
        return self.state["r"]

    def get_velocities(self):
        return self.state["u"]

    def select_case(self, case_name, state=None):
        self.case_manager.select(case_name, state)
        self.solver_manager.is_solver_initialized = False
        self.set_dt()
        self.reset_scene()

    def select_solver(self, solver_name):
        self.solver_manager.select(solver_name)
        self.solver_manager.is_solver_initialized = False
        self.set_dt()
        self.reset_scene()

    def reset_scene(self):
        self.curr_timestamp = 0.0
        self.step = 0
        self.state = jax.tree_util.tree_map(lambda x: jnp.array(x), self.case_manager.state)
        self.set_save_directory()

    def advance(self):
        self.state = self.solver_manager.next(self.case_manager, self.step, self.state)
        self.curr_timestamp += self.dt
        self.step += 1

    def get_current_save_directory(self):
        return self.curr_save_directory
    
    def get_timestamp(self):
        return self.curr_timestamp