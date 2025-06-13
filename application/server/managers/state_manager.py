from application.server.managers.case_manager import CaseManager
from application.server.managers.solver_manager import SolverManager
import jax
import jax.numpy as jnp

class StateManager:
    def __init__(self):
        self.case_manager = CaseManager()
        self.solver_manager = SolverManager()
        
        self.case_manager.select("db")
        self.solver_manager.select("wcsph")
        # self.solver_manager.init_solver(self.case_manager)
        self.dt = self.case_manager.cfg.solver.dt
        self.reset_scene()

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

    def select_case(self, case_name):
        self.case_manager.select(case_name)
        # self.solver_manager.init_solver(self.case_manager)
        self.dt = self.case_manager.cfg.solver.dt
        self.solver_manager.is_solver_initialized = False
        self.reset_scene()

    def select_solver(self, solver_name):
        self.solver_manager.select(solver_name)
        # self.solver_manager.init_solver(self.case_manager)
        self.reset_scene()

    def reset_scene(self):
        self.step = 0
        self.state = jax.tree_util.tree_map(lambda x: jnp.array(x), self.case_manager.state)

    def advance(self):
        self.state = self.solver_manager.next(self.case_manager, self.step, self.state)
        self.step += 1
