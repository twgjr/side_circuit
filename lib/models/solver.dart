import 'dart:html';

import 'expression.dart';
import 'model.dart';


enum SolverState { Unknown, Sat, Unsat }

class Solver {
  Order order = Order();
  Model model;

  Solver(this.model);

  int solve() {
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

  // void simplify(){
  //   List<Expression> unvisited = [];
  //
  //   // start by spreading out constants for simple solutions
  //   for(Expression constant in model.constants){
  //     unvisited.add(constant.parents[0]);  //constants have only one parent no children
  //   }
  //
  //   // search by variable list instead if no constants were defined (unlikely corner case)
  //   if(model.constants.isEmpty){
  //     for(Expression variable in model.variables){
  //       for(Expression parent in variable.parents)
  //       unvisited.add(parent);  //variables have multiple parents no children
  //     }
  //   }
  //   // propagate ranges throughout the model
  //   while(unvisited.isNotEmpty){
  //
  //   }
  // }

  bool decide() {
    List<Expression> unvisitedVars = [];
    List<Expression> visited = [
    ]; // order any expressions were visited, including variables
    Expression destination; // points to the current expression being inspected
    //  to track the visited variables that need to be revisited if expression
    //  cannot be set (unsat)
    Expression unSatExpr;


    // create a new list that points to all the variables to be visited
    for (Expression variable in model.variables) {
      unvisitedVars.add(variable);
    }

    // set the destination to point to the first variable
    if (unvisitedVars.isNotEmpty) {
      destination = unvisitedVars.first;
    } else {
      return false; // no variables, nothing to solve
    }

    while (unvisitedVars.isNotEmpty) {
      // if there are any unvisited children, that means more variables need to
      // be explored before continuing up the tree
      for (Expression child in destination.children) {
        if (!child.isVisited) {
          destination = unvisitedVars.first;
          unvisitedVars.removeAt(0);
        }
      }

      // all of destination's children have been visited, destination ready to be set
      visited.add(destination);
      if (!destination.setValue()) {
        unSatExpr = destination; // remember the expression that was unsat

        // starting with the destination that was unsat, set its children to make it sat

        // get list of children of unsat expression in order of visited
        List<Expression> unSatChildren = [];
        for (Expression expr in visited) {
          for(Expression child in unSatExpr.children) {
            if(expr == child) {
              unSatChildren.add(child);
            }
          }
        }

        //@todo add function to set values and/or ranges of unsat children in
        //@todo starting with most recent visited
        //@todo this would be opportunity to use some recursion to split/double ranges
        //@todo of the children of unsatExpr.
        //@todo would need to include a counter for max iterations
        //@todo should always first check all combinations of boundaries of the range for children
        // @todo of the unSat expression
        //@todo whichever combination works first or best will then propagate down the tree to set
        //@todo affected children.
      }
    }

    // attempt to keep going up the expression tree towards the root
    for (Expression parent in destination.parents) {
      if (!parent.isVisited) {
        destination = parent;
      }
    }
  }
}}
