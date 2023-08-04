import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/view/visual.dart';

class Node {
  Circuit circuit;
  List<Terminal> terminals = [];
  Visual visual = Visual();
  int id;

  Node(this.circuit) : id = circuit.maxNodeId() + 1;

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
