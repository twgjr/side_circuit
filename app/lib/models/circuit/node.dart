import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/terminal.dart';

class Node {
  Circuit circuit;
  List<Terminal> terminals = [];

  Node(this.circuit);

  int numTerminals() => terminals.length;
  int get index => circuit.nodes.indexOf(this);

  void addTerminal(Terminal terminal) {
    if (!terminals.contains(terminal)) {
      terminals.add(terminal);
    }
  }

  void removeTerminal(Terminal terminal) {
    if (terminals.contains(terminal)) {
      terminals.remove(terminal);
    }
  }
}
