import 'expression.dart';
import 'model.dart';


enum SolverState { Unknown, Sat, Unsat }

class Solver {
  Order order = Order();
  Model model;

  Solver(this.model);

  void sortVariables() {
    // sort variables in order of:
    // - most constrained (smallest range of possible solutions)
    // - most occurrences (number of times appearing in equations
  }

  void sortExpressions() {
    // sort expressions (roots) in order of:
    // - containing the mose constrained variables
    // - most occurrences (number of times appearing in equations
  }

  void simplify() {
    // grab low hanging fruit for easy solutions:
    // - solve simple single level expressions such as 'x = 1' or 'y > 2' or 'pi/2 = sin(theta)'
    // - continue up the tree
  }

  int solve() {
    sortVariables();
    sortExpressions();
    simplify();

    bool decided = false;
    var solverState = SolverState.Unknown;
    while (!decided) {
      switch (solverState) {
        case SolverState.Unknown:
          {
            //solverState = check() ? SolverState.Sat : SolverState.Unsat;
            break;
          }
        case SolverState.Sat:
          {
            print("Solution is sat.");
            decided = true;
            break;
          }
        case SolverState.Unsat:
          {
            print("Solution is UNSAT!");
            decided = true;
            break;
          }
      }
    }
    return solverState.index;
  }
}
