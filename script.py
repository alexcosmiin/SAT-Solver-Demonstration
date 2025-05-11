import copy
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import sys
import threading
import multiprocessing

def make_dpll_random(verbose):
    return DPLLSolver(verbose=verbose, branching_heuristic="random")

def make_dpll_most_common(verbose):
    return DPLLSolver(verbose=verbose, branching_heuristic="most_common")

def make_dpll_jeroslow_wang(verbose):
    return DPLLSolver(verbose=verbose, branching_heuristic="jeroslow_wang")

class SolverThread(threading.Thread):
    def __init__(self, solver, problem):
        super().__init__()
        self.solver = solver
        self.problem = problem
        self.result = None
        self.exception = None

    def run(self):
        try:
            self.result = self.solver.solve(self.problem)
        except Exception as e:
            self.exception = e

class SATInstance:
    """Representation of a SAT formula in CNF form."""

    def __init__(self, clauses=None, num_vars=0):
        """
        Initialize a SAT instance.

        Args:
            clauses: List of clauses, where each clause is a list of literals.
                    A literal is a positive or negative integer representing a variable.
            num_vars: Number of variables in the formula.
        """
        self.clauses = clauses if clauses is not None else []
        self.num_vars = num_vars

    def from_dimacs(self, filename):
        """
        Load a SAT instance from a DIMACS CNF file.

        Args:
            filename: Path to the DIMACS CNF file.
        """
        self.clauses = []
        self.num_vars = 0

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue

                if line.startswith('p cnf'):
                    parts = line.split()
                    self.num_vars = int(parts[2])
                    continue

                clause = [int(x) for x in line.split() if x != '0']
                if clause:  # Skip empty clauses
                    self.clauses.append(clause)

        return self

    def generate_random(self, num_vars, num_clauses, clause_length=3):
        """
        Generate a random SAT instance.

        Args:
            num_vars: Number of variables.
            num_clauses: Number of clauses.
            clause_length: Length of each clause.
        """
        self.num_vars = num_vars
        self.clauses = []

        variables = list(range(1, num_vars + 1))

        for _ in range(num_clauses):
            clause = []
            selected_vars = random.sample(variables, min(clause_length, num_vars))

            for var in selected_vars:
                literal = var if random.random() < 0.5 else -var
                clause.append(literal)

            self.clauses.append(clause)

        return self

    def __str__(self):
        """String representation of the SAT instance."""
        result = f"SAT instance with {self.num_vars} variables and {len(self.clauses)} clauses:\n"
        for i, clause in enumerate(self.clauses):
            result += f"Clause {i + 1}: {clause}\n"
        return result

    def display_formula(self):
        """Display the formula in a more readable format."""
        result = ""
        for i, clause in enumerate(self.clauses):
            result += "("
            for j, lit in enumerate(clause):
                if lit > 0:
                    result += f"x{abs(lit)}"
                else:
                    result += f"¬x{abs(lit)}"
                if j < len(clause) - 1:
                    result += " ∨ "
            result += ")"
            if i < len(self.clauses) - 1:
                result += " ∧ "
        return result


class ResolutionSolver:
    """
    Implementation of the Resolution algorithm for SAT solving.
    Resolution is primarily used as a proof system for unsatisfiability.
    """

    def __init__(self, verbose=True):
        """
        Initialize the Resolution solver.

        Args:
            verbose: Whether to print detailed information during solving.
        """
        self.verbose = verbose
        self.stats = {"resolutions": 0, "tautologies_removed": 0, "subsumed_clauses": 0}

    def resolve(self, clause1, clause2, var):
        """
        Perform resolution on two clauses with respect to a variable.

        Args:
            clause1, clause2: The two clauses to resolve.
            var: The variable to resolve on.

        Returns:
            The resolvent clause or None if the clauses cannot be resolved on var.
        """
        if var in clause1 and -var in clause2:
            c1_without_var = [lit for lit in clause1 if lit != var]
            c2_without_var = [lit for lit in clause2 if lit != -var]
        elif -var in clause1 and var in clause2:
            c1_without_var = [lit for lit in clause1 if lit != -var]
            c2_without_var = [lit for lit in clause2 if lit != var]
        else:
            return None

        resolvent = sorted(list(set(c1_without_var + c2_without_var)))

        # Check if resolvent is a tautology (contains both a literal and its negation)
        for lit in resolvent:
            if -lit in resolvent:
                if self.verbose:
                    print(f"  Resolvent {resolvent} is a tautology, skipping")
                self.stats["tautologies_removed"] += 1
                return None

        if self.verbose:
            print(f"  Resolving on variable {var}:")
            print(f"    Clause 1: {clause1}")
            print(f"    Clause 2: {clause2}")
            print(f"    Resolvent: {resolvent}")

        self.stats["resolutions"] += 1
        return resolvent

    def is_subsumed(self, clause, clause_set):
        """
        Check if a clause is subsumed by any clause in the clause set.
        A clause C1 subsumes C2 if C1 is a subset of C2.

        Args:
            clause: The clause to check.
            clause_set: The set of clauses.

        Returns:
            True if the clause is subsumed, False otherwise.
        """
        for c in clause_set:
            if set(c).issubset(set(clause)) and c != clause:
                if self.verbose:
                    print(f"  Clause {clause} is subsumed by {c}, skipping")
                self.stats["subsumed_clauses"] += 1
                return True
        return False

    def solve(self, sat_instance, max_steps=1000):
        """
        Solve a SAT instance using resolution.

        Args:
            sat_instance: The SAT instance to solve.
            max_steps: Maximum number of resolution steps to perform.

        Returns:
            "UNSATISFIABLE" if the empty clause is derived,
            "UNKNOWN" if the maximum steps are reached without finding a contradiction.
        """
        if self.verbose:
            print("\n=== Resolution Solver ===")
            print(f"Starting with {len(sat_instance.clauses)} clauses")

        # Reset statistics
        self.stats = {"resolutions": 0, "tautologies_removed": 0, "subsumed_clauses": 0}

        # Make a copy of the clauses to avoid modifying the original instance
        clauses = [sorted(clause) for clause in sat_instance.clauses]

        # Check for empty clause in the initial set
        if [] in clauses:
            if self.verbose:
                print("Empty clause found in initial formula, formula is UNSATISFIABLE")
            return "UNSATISFIABLE"

        # Main resolution loop
        steps = 0
        while steps < max_steps:
            new_clauses = []

            # Try to resolve each pair of clauses
            for i in range(len(clauses)):
                for j in range(i + 1, len(clauses)):
                    c1 = clauses[i]
                    c2 = clauses[j]

                    # Find variables that appear in both clauses with opposite polarity
                    c1_vars = set(abs(lit) for lit in c1)
                    c2_vars = set(abs(lit) for lit in c2)
                    common_vars = c1_vars.intersection(c2_vars)

                    for var in common_vars:
                        if (var in c1 and -var in c2) or (-var in c1 and var in c2):
                            resolvent = self.resolve(c1, c2, var if var in c1 else -var)

                            # If resolvent is empty, we've derived a contradiction
                            if resolvent is not None:
                                if not resolvent:
                                    if self.verbose:
                                        print("Empty clause derived, formula is UNSATISFIABLE")
                                    return "UNSATISFIABLE"

                                # Check if resolvent is new and not subsumed
                                if resolvent not in clauses and resolvent not in new_clauses and not self.is_subsumed(
                                        resolvent, clauses + new_clauses):
                                    new_clauses.append(resolvent)

            # If no new clauses were derived, we've reached saturation
            if not new_clauses:
                if self.verbose:
                    print("Resolution saturation reached without finding contradiction")
                    print(f"Final clause count: {len(clauses)}")
                    print(f"Statistics: {self.stats}")
                return "UNKNOWN"

            if self.verbose:
                print(f"Step {steps + 1}: {len(new_clauses)} new clauses derived")

            clauses.extend(new_clauses)
            steps += 1

        if self.verbose:
            print(f"Maximum steps ({max_steps}) reached without finding contradiction")
            print(f"Final clause count: {len(clauses)}")
            print(f"Statistics: {self.stats}")

        return "UNKNOWN"


class DPSolver:
    """
    Implementation of the Davis-Putnam (DP) algorithm for SAT solving.
    DP uses resolution to eliminate variables systematically.
    """

    def __init__(self, verbose=True):
        """
        Initialize the DP solver.

        Args:
            verbose: Whether to print detailed information during solving.
        """
        self.verbose = verbose
        self.stats = {"variables_eliminated": 0, "unit_propagations": 0, "pure_literals": 0}

    def count_occurrences(self, clauses):
        """
        Count occurrences of each literal in the clause set.

        Args:
            clauses: List of clauses.

        Returns:
            Dictionary mapping literals to their occurrence count.
        """
        occurrences = defaultdict(int)
        for clause in clauses:
            for lit in clause:
                occurrences[lit] += 1
        return occurrences

    def unit_propagation(self, clauses):
        """
        Apply unit propagation to simplify the formula.

        Args:
            clauses: List of clauses.

        Returns:
            Tuple (simplified clauses, result), where result is:
            - "SATISFIABLE" if all clauses are satisfied
            - "UNSATISFIABLE" if a contradiction is found
            - None if no conclusion can be drawn
        """
        change = True
        unit_clauses = [clause[0] for clause in clauses if len(clause) == 1]

        while change and unit_clauses:
            change = False
            unit = unit_clauses.pop(0)

            if self.verbose:
                print(f"  Unit propagation: assigning {unit}")

            self.stats["unit_propagations"] += 1

            # Remove clauses containing the unit literal
            new_clauses = []
            for clause in clauses:
                if unit in clause:
                    continue  # Clause is satisfied

                # Remove negation of unit from clauses
                if -unit in clause:
                    new_clause = [lit for lit in clause if lit != -unit]
                    if not new_clause:
                        if self.verbose:
                            print("  Empty clause derived, formula is UNSATISFIABLE")
                        return [], "UNSATISFIABLE"
                    new_clauses.append(new_clause)
                else:
                    new_clauses.append(clause)

            # Check if we derived new unit clauses
            clauses = new_clauses
            new_units = [clause[0] for clause in clauses if
                         len(clause) == 1 and clause[0] not in unit_clauses and -clause[0] not in unit_clauses]

            if new_units:
                unit_clauses.extend(new_units)
                change = True

        if not clauses:
            if self.verbose:
                print("  All clauses satisfied")
            return [], "SATISFIABLE"

        return clauses, None

    def pure_literal_elimination(self, clauses):
        """
        Apply pure literal elimination to simplify the formula.

        Args:
            clauses: List of clauses.

        Returns:
            Simplified clauses.
        """
        if not clauses:
            return []

        # Count occurrences of each literal
        occurrences = self.count_occurrences(clauses)

        # Find pure literals
        pure_literals = []
        for lit in occurrences:
            if -lit not in occurrences:
                pure_literals.append(lit)
                if self.verbose:
                    print(f"  Pure literal elimination: {lit} is pure")
                self.stats["pure_literals"] += 1

        if not pure_literals:
            return clauses

        # Remove clauses containing pure literals
        new_clauses = []
        for clause in clauses:
            if any(lit in pure_literals for lit in clause):
                continue  # Clause is satisfied
            new_clauses.append(clause)

        return new_clauses

    def eliminate_variable(self, clauses, var):
        """
        Eliminate a variable from the formula using resolution.

        Args:
            clauses: List of clauses.
            var: Variable to eliminate.

        Returns:
            Formula with var eliminated.
        """
        if self.verbose:
            print(f"  Eliminating variable {var}")

        self.stats["variables_eliminated"] += 1

        # Separate clauses containing var and -var
        pos_clauses = [clause for clause in clauses if var in clause]
        neg_clauses = [clause for clause in clauses if -var in clause]
        rest_clauses = [clause for clause in clauses if var not in clause and -var not in clause]

        # Perform resolution on all pairs of pos_clauses and neg_clauses
        resolvents = []
        for pos_clause in pos_clauses:
            for neg_clause in neg_clauses:
                # Create resolvent
                resolvent = [lit for lit in pos_clause if lit != var] + [lit for lit in neg_clause if lit != -var]
                # Remove duplicates and check for tautology
                resolvent = list(set(resolvent))

                is_tautology = False
                for lit in resolvent:
                    if -lit in resolvent:
                        is_tautology = True
                        break

                if not is_tautology and resolvent not in resolvents:
                    resolvents.append(resolvent)

        return rest_clauses + resolvents

    def select_variable(self, clauses):
        """
        Select a variable to eliminate.

        Args:
            clauses: List of clauses.

        Returns:
            The variable to eliminate.
        """
        # Count occurrences of each variable
        var_counts = defaultdict(int)
        for clause in clauses:
            for lit in clause:
                var_counts[abs(lit)] += 1

        # Select the variable with the minimum occurrences
        if not var_counts:
            return None
        return min(var_counts.items(), key=lambda x: x[1])[0]

    def solve(self, sat_instance):
        """
        Solve a SAT instance using the Davis-Putnam algorithm.

        Args:
            sat_instance: The SAT instance to solve.

        Returns:
            "SATISFIABLE", "UNSATISFIABLE", or "UNKNOWN".
        """
        if self.verbose:
            print("\n=== Davis-Putnam Solver ===")

        # Reset statistics
        self.stats = {"variables_eliminated": 0, "unit_propagations": 0, "pure_literals": 0}

        # Make a copy of the clauses to avoid modifying the original instance
        clauses = [list(clause) for clause in sat_instance.clauses]

        if self.verbose:
            print(f"Starting with {len(clauses)} clauses and {sat_instance.num_vars} variables")

        # Check for empty clause in the initial set
        if [] in clauses:
            if self.verbose:
                print("Empty clause found in initial formula, formula is UNSATISFIABLE")
            return "UNSATISFIABLE"

        # Apply unit propagation and pure literal elimination
        clauses, result = self.unit_propagation(clauses)
        if result:
            return result

        clauses = self.pure_literal_elimination(clauses)
        if not clauses:
            return "SATISFIABLE"

        # Main DP loop
        while True:
            # Select a variable to eliminate
            var = self.select_variable(clauses)
            if not var:
                if self.verbose:
                    print("All variables eliminated, formula is SATISFIABLE")
                    print(f"Statistics: {self.stats}")
                return "SATISFIABLE"

            # Eliminate the variable
            clauses = self.eliminate_variable(clauses, var)

            # Check for empty clause
            if [] in clauses:
                if self.verbose:
                    print("Empty clause derived, formula is UNSATISFIABLE")
                    print(f"Statistics: {self.stats}")
                return "UNSATISFIABLE"

            # Apply unit propagation and pure literal elimination
            clauses, result = self.unit_propagation(clauses)
            if result:
                if self.verbose:
                    print(f"Statistics: {self.stats}")
                return result

            clauses = self.pure_literal_elimination(clauses)
            if not clauses:
                if self.verbose:
                    print("All clauses satisfied, formula is SATISFIABLE")
                    print(f"Statistics: {self.stats}")
                return "SATISFIABLE"

            if self.verbose:
                print(f"  Current clause count: {len(clauses)}")


class DPLLSolver:
    """
    Implementation of the Davis-Putnam-Logemann-Loveland (DPLL) algorithm for SAT solving.
    DPLL is a backtracking-based search algorithm that extends DP.
    """

    def __init__(self, verbose=True, branching_heuristic="random"):
        """
        Initialize the DPLL solver.

        Args:
            verbose: Whether to print detailed information during solving.
            branching_heuristic: Strategy for variable selection ("random", "most_common", "jeroslow_wang").
        """
        self.verbose = verbose
        self.branching_heuristic = branching_heuristic
        self.stats = {
            "decisions": 0,
            "backtracks": 0,
            "unit_propagations": 0,
            "pure_literals": 0,
            "conflicts": 0
        }
        self.assignment = {}
        self.decision_level = 0

    def count_occurrences(self, clauses):
        """
        Count occurrences of each literal in the clause set.

        Args:
            clauses: List of clauses.

        Returns:
            Dictionary mapping literals to their occurrence count.
        """
        occurrences = defaultdict(int)
        for clause in clauses:
            for lit in clause:
                occurrences[lit] += 1
        return occurrences

    def unit_propagation(self, clauses, assignment):
        """
        Apply unit propagation to simplify the formula.

        Args:
            clauses: List of clauses.
            assignment: Current partial assignment.

        Returns:
            Tuple (simplified clauses, assignment, result), where result is:
            - "CONFLICT" if a conflict is found
            - None if no conflict is found
        """
        change = True
        unit_clauses = [clause[0] for clause in clauses if len(clause) == 1]

        while change and unit_clauses:
            change = False
            unit = unit_clauses.pop(0)

            # Check if this unit contradicts the current assignment
            if -unit in assignment:
                if self.verbose:
                    print(f"  Conflict: unit {unit} contradicts assignment {-unit}")
                self.stats["conflicts"] += 1
                return clauses, assignment, "CONFLICT"

            # Add unit to assignment if not already assigned
            if unit not in assignment:
                if self.verbose:
                    print(f"  Unit propagation: assigning {unit}")
                assignment[unit] = True
                self.stats["unit_propagations"] += 1

            # Apply the assignment to simplify clauses
            new_clauses = []
            for clause in clauses:
                # Skip satisfied clauses
                if any(lit in assignment for lit in clause):
                    continue

                # Remove falsified literals
                new_clause = [lit for lit in clause if -lit not in assignment]

                # If clause is empty, we have a conflict
                if not new_clause:
                    if self.verbose:
                        print("  Conflict: empty clause derived")
                    self.stats["conflicts"] += 1
                    return clauses, assignment, "CONFLICT"

                new_clauses.append(new_clause)

            clauses = new_clauses

            # Find new unit clauses
            new_units = [clause[0] for clause in clauses if
                         len(clause) == 1 and clause[0] not in unit_clauses and -clause[0] not in unit_clauses]

            if new_units:
                unit_clauses.extend(new_units)
                change = True

        return clauses, assignment, None

    def pure_literal_elimination(self, clauses, assignment):
        """
        Apply pure literal elimination to simplify the formula.

        Args:
            clauses: List of clauses.
            assignment: Current partial assignment.

        Returns:
            Tuple (simplified clauses, assignment).
        """
        if not clauses:
            return clauses, assignment

        # Count occurrences of each literal
        occurrences = self.count_occurrences(clauses)

        # Find pure literals
        pure_literals = []
        for lit in occurrences:
            if -lit not in occurrences and lit not in assignment and -lit not in assignment:
                pure_literals.append(lit)
                if self.verbose:
                    print(f"  Pure literal elimination: {lit} is pure")
                assignment[lit] = True
                self.stats["pure_literals"] += 1

        if not pure_literals:
            return clauses, assignment

        # Remove clauses containing pure literals
        new_clauses = []
        for clause in clauses:
            if any(lit in pure_literals for lit in clause):
                continue  # Clause is satisfied
            new_clauses.append(clause)

        return new_clauses, assignment

    def select_variable(self, clauses, assignment):
        """
        Select an unassigned variable according to the branching heuristic.

        Args:
            clauses: List of clauses.
            assignment: Current partial assignment.

        Returns:
            Variable to branch on.
        """
        # Find unassigned variables
        all_vars = set()
        for clause in clauses:
            for lit in clause:
                all_vars.add(abs(lit))

        unassigned = [var for var in all_vars if var not in assignment and -var not in assignment]

        if not unassigned:
            return None

        if self.branching_heuristic == "random":
            return random.choice(unassigned)

        elif self.branching_heuristic == "most_common":
            # Count occurrences of each variable
            var_counts = Counter()
            for clause in clauses:
                for lit in clause:
                    var = abs(lit)
                    if var in unassigned:
                        var_counts[var] += 1

            # Select the most common variable
            return var_counts.most_common(1)[0][0] if var_counts else random.choice(unassigned)

        elif self.branching_heuristic == "jeroslow_wang":
            # Jeroslow-Wang heuristic gives more weight to literals in shorter clauses
            scores = defaultdict(float)
            for clause in clauses:
                for lit in clause:
                    var = abs(lit)
                    if var in unassigned:
                        scores[var] += 2 ** -len(clause)

            # Select the variable with the highest score
            return max(scores.items(), key=lambda x: x[1])[0] if scores else random.choice(unassigned)

        else:
            return random.choice(unassigned)

    def dpll(self, clauses, assignment):
        """
        Recursive DPLL procedure.

        Args:
            clauses: List of clauses.
            assignment: Current partial assignment.

        Returns:
            "SATISFIABLE" with a satisfying assignment or "UNSATISFIABLE".
        """
        # Apply unit propagation
        clauses, assignment, conflict = self.unit_propagation(clauses, assignment)
        if conflict:
            return "UNSATISFIABLE"

        # All clauses satisfied?
        if not clauses:
            self.assignment = assignment
            return "SATISFIABLE"

        # Apply pure literal elimination
        clauses, assignment = self.pure_literal_elimination(clauses, assignment)
        if not clauses:
            self.assignment = assignment
            return "SATISFIABLE"

        # Select a variable to branch on
        var = self.select_variable(clauses, assignment)
        if not var:
            self.assignment = assignment
            return "SATISFIABLE"

        # Try var = True
        self.stats["decisions"] += 1
        self.decision_level += 1
        if self.verbose:
            print(f"Decision {self.stats['decisions']} (level {self.decision_level}): try {var}")

        assignment_copy = assignment.copy()
        assignment_copy[var] = True
        result = self.dpll(clauses, assignment_copy)

        if result == "SATISFIABLE":
            return "SATISFIABLE"

        # Try var = False
        self.stats["backtracks"] += 1
        if self.verbose:
            print(f"Backtrack {self.stats['backtracks']}: try {-var}")

        assignment_copy = assignment.copy()
        assignment_copy[-var] = True
        result = self.dpll(clauses, assignment_copy)

        self.decision_level -= 1
        return result

    def solve(self, sat_instance):
        """
        Solve a SAT instance using the DPLL algorithm.

        Args:
            sat_instance: The SAT instance to solve.

        Returns:
            "SATISFIABLE" or "UNSATISFIABLE".
        """
        if self.verbose:
            print(f"\n=== DPLL Solver (heuristic: {self.branching_heuristic}) ===")

        # Reset statistics and state
        self.stats = {
            "decisions": 0,
            "backtracks": 0,
            "unit_propagations": 0,
            "pure_literals": 0,
            "conflicts": 0
        }
        self.assignment = {}
        self.decision_level = 0

        # Make a copy of the clauses to avoid modifying the original instance
        clauses = [list(clause) for clause in sat_instance.clauses]

        if self.verbose:
            print(f"Starting with {len(clauses)} clauses and {sat_instance.num_vars} variables")

        # Check for empty clause in the initial set
        if [] in clauses:
            if self.verbose:
                print("Empty clause found in initial formula, formula is UNSATISFIABLE")
            return "UNSATISFIABLE"

        # Run DPLL
        result = self.dpll(clauses, {})

        if self.verbose:
            print(f"Result: {result}")
            if result == "SATISFIABLE":
                print("Satisfying assignment:")
                for var in range(1, sat_instance.num_vars + 1):
                    if var in self.assignment:
                        print(f"  x{var} = True")
                    elif -var in self.assignment:
                        print(f"  x{var} = False")
                    else:
                        print(f"  x{var} = True or False (free variable)")
            print(f"Statistics: {self.stats}")

        return result

def solver_worker(algo_constructor, problem, verbose, return_dict):
    try:
        solver = algo_constructor(verbose)
        result = solver.solve(problem)
        return_dict["result"] = result
    except Exception as e:
        return_dict["exception"] = str(e)

def run_experiment(problem_sets, algorithms, timeout=60, verbose=False):
    """
    Run experiments comparing different SAT solving algorithms on various problem sets.

    Args:
        problem_sets: List of SAT instances to solve.
        algorithms: Dictionary of SAT solving algorithm constructors.
        timeout: Timeout per instance in seconds.
        verbose: Verbose output.

    Returns:
        Dictionary with results.
    """
    results = {
        "solving_times": defaultdict(list),
        "results": defaultdict(list),
        "timeouts": defaultdict(int)
    }

    def worker(algo_constructor, problem, verbose, return_dict):
        try:
            solver = algo_constructor(verbose)
            result = solver.solve(problem)
            return_dict["result"] = result
        except Exception as e:
            return_dict["exception"] = str(e)

    for i, problem in enumerate(problem_sets):
        print(f"\n======= Problem {i + 1}/{len(problem_sets)} =======")
        print(f"Number of variables: {problem.num_vars}")
        print(f"Number of clauses: {len(problem.clauses)}")

        formula_str = problem.display_formula()
        preview_len = min(100, len(formula_str))
        print(f"Formula preview: {formula_str[:preview_len]}..." if len(formula_str) > preview_len else formula_str)

        for algo_name, algo_constructor in algorithms.items():
            print(f"\nRunning {algo_name}...")

            manager = multiprocessing.Manager()
            return_dict = manager.dict()

            process = multiprocessing.Process(target=solver_worker, args=(algo_constructor, problem, verbose, return_dict))

            start_time = time.perf_counter()
            process.start()
            process.join(timeout)
            elapsed_time = time.perf_counter() - start_time

            if process.is_alive():
                process.terminate()
                process.join()
                print(f"{algo_name} timed out after {elapsed_time:.5f} seconds")
                results["timeouts"][algo_name] += 1
                results["solving_times"][algo_name].append(timeout)
                results["results"][algo_name].append("TIMEOUT")
            elif "exception" in return_dict:
                print(f"{algo_name} encountered an error: {return_dict['exception']}")
                results["solving_times"][algo_name].append(None)
                results["results"][algo_name].append("ERROR")
            else:
                print(f"{algo_name} completed in {elapsed_time:.5f} seconds with result: {return_dict['result']}")
                results["solving_times"][algo_name].append(elapsed_time)
                results["results"][algo_name].append(return_dict["result"])

    return results


def plot_results(results, algorithms, title="Algorithm Performance Comparison"):
    """
    Plot the results of the experiments.

    Args:
        results: Dictionary with experiment results.
        algorithms: Dictionary of algorithm names to solver objects.
        title: Title of the plot.
    """
    plt.figure(figsize=(12, 6))

    # Create performance plot
    algo_names = list(algorithms.keys())
    solving_times = [results["solving_times"][algo] for algo in algo_names]

    plt.subplot(1, 2, 1)
    plt.boxplot(solving_times, tick_labels=algo_names)
    plt.title("Solving Time Distribution")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Create success rate plot
    success_rates = []
    for algo in algo_names:
        total_problems = len(results["results"][algo])
        timeouts = results["timeouts"].get(algo, 0)
        success_rate = (total_problems - timeouts) / total_problems * 100 if total_problems > 0 else 0
        success_rates.append(success_rate)

    plt.subplot(1, 2, 2)
    plt.bar(algo_names, success_rates)
    plt.title("Success Rate")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 100)

    plt.suptitle(title)
    plt.tight_layout()

    return plt


def generate_problem_sets(num_instances=5, min_vars=10, max_vars=20, clause_ratio=4.3):
    """
    Generate a set of random SAT problems.

    Args:
        num_instances: Number of problem instances to generate.
        min_vars: Minimum number of variables per instance.
        max_vars: Maximum number of variables per instance.
        clause_ratio: Ratio of clauses to variables.

    Returns:
        List of SAT instances.
    """
    problem_sets = []

    for _ in range(num_instances):
        num_vars = random.randint(min_vars, max_vars)
        num_clauses = int(num_vars * clause_ratio)

        sat_instance = SATInstance().generate_random(num_vars, num_clauses)
        problem_sets.append(sat_instance)

    return problem_sets


def main():
    """Main function to demonstrate SAT solvers."""

    print("=== SAT Solver Demonstration ===")

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='SAT Solver Comparison')
    parser.add_argument('--mode', choices=['demo', 'experiment'], default='demo',
                        help='Run mode: demo or experiment')
    parser.add_argument('--instances', type=int, default=5,
                        help='Number of random instances for experiment mode')
    parser.add_argument('--min_vars', type=int, default=10,
                        help='Minimum number of variables per instance')
    parser.add_argument('--max_vars', type=int, default=20,
                        help='Maximum number of variables per instance')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout in seconds for each algorithm on each instance')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--file', type=str, default=None,
                        help='DIMACS CNF file to solve (for demo mode)')

    args = parser.parse_args()

    if args.mode == 'demo':
        # Create a SAT instance
        if args.file:
            sat_instance = SATInstance().from_dimacs(args.file)
            print(f"Loaded SAT instance from {args.file}")
        else:
            # Generate a random SAT instance
            num_vars = 8
            num_clauses = 12
            sat_instance = SATInstance().generate_random(num_vars, num_clauses)
            print(f"Generated random SAT instance with {num_vars} variables and {num_clauses} clauses")

        print("\nFormula:")
        print(sat_instance.display_formula())

        # Solve with different algorithms
        print("\nSolving with Resolution...")
        resolution_solver = ResolutionSolver(verbose=args.verbose)
        resolution_result = resolution_solver.solve(sat_instance)
        print(f"Resolution result: {resolution_result}")

        print("\nSolving with Davis-Putnam...")
        dp_solver = DPSolver(verbose=args.verbose)
        dp_result = dp_solver.solve(sat_instance)
        print(f"Davis-Putnam result: {dp_result}")

        print("\nSolving with DPLL (random branching)...")
        dpll_solver = DPLLSolver(verbose=args.verbose, branching_heuristic="random")
        dpll_result = dpll_solver.solve(sat_instance)
        print(f"DPLL result: {dpll_result}")

        print("\nSolving with DPLL (most common branching)...")
        dpll_mc_solver = DPLLSolver(verbose=args.verbose, branching_heuristic="most_common")
        dpll_mc_result = dpll_mc_solver.solve(sat_instance)
        print(f"DPLL (most common) result: {dpll_mc_result}")

    elif args.mode == 'experiment':
        print(f"Running experiment with {args.instances} random instances...")
        print(f"Variable range: {args.min_vars} to {args.max_vars}")
        print(f"Timeout: {args.timeout} seconds per algorithm")

        # Generate problem sets
        problem_sets = generate_problem_sets(
            num_instances=args.instances,
            min_vars=args.min_vars,
            max_vars=args.max_vars
        )

        # Define algorithms
        algorithms = {
            "Resolution": ResolutionSolver,
            "Davis-Putnam": DPSolver,
            "DPLL (Random)": make_dpll_random,
            "DPLL (Most Common)": make_dpll_most_common,
            "DPLL (Jeroslow-Wang)": make_dpll_jeroslow_wang,
        }

        # Run experiments
        results = run_experiment(problem_sets, algorithms, timeout=args.timeout, verbose=args.verbose)

        # Plot results
        plot = plot_results(results, algorithms)
        plot.savefig("sat_solver_comparison.png")
        print("Results saved to sat_solver_comparison.png")

        # Print summary
        print("\nExperiment Summary:")
        for algo in algorithms:
            results_list = results["results"].get(algo, [])
            times_list = results["solving_times"].get(algo, [])
            total = len(results_list)
            timeouts = results["timeouts"].get(algo, 0)
            sat_count = results_list.count("SATISFIABLE")
            unsat_count = results_list.count("UNSATISFIABLE")
            non_timeout_times = [t for t in times_list if t is not None]

            print(f"{algo}:")
            if total > 0:
                success_rate = (total - timeouts) / total * 100
                avg_time = np.mean(non_timeout_times) if non_timeout_times else 0
                print(f"  Success rate: {success_rate:.1f}%")
                print(f"  SATISFIABLE: {sat_count}")
                print(f"  UNSATISFIABLE: {unsat_count}")
                print(f"  Timeouts: {timeouts}")
                print(f"  Average time: {avg_time:.2f} seconds")
            else:
                print("  No results recorded.")


if __name__ == "__main__":
    main()

