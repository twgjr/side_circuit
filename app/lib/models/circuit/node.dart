import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/wire.dart';
import 'package:app/models/view/visual.dart';

class Node {
  Circuit circuit;
  List<Wire> wires = [];
  Visual visual = Visual();
  int id;

  Node(this.circuit) : id = circuit.maxNodeId() + 1;

  void addWire(Wire wire) {
    if (!wires.contains(wire)) {
      wires.add(wire);
    }
  }

  void removeWire(Wire wire) {
    wires.remove(wire);
  }

  Node copyWith({Circuit? circuit}) {
    final node = Node(circuit ?? this.circuit);
    node.wires = [];
    for (Wire wire in wires) {
      node.wires.add(wire.copyWith(node: node));
    }
    node.visual = visual.copy();
    return node;
  }
}
