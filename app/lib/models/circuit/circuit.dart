/*
CIRCUITS is a library for creating and analyzing electrical circuits.  It primarily
follows the same structure as the SPICE circuit simulator.  It can access the ngspice 
API.
*/

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/models/circuit/terminal.dart';

enum Quantity { I, V, P }

class Circuit {
  List<Node> nodes = [];
  List<Device> elements = [];

  Circuit();

  bool isEmpty() => elements.isEmpty && nodes.isEmpty;

  void addElement(Device element) => elements.add(element);
  void addElementOf(DeviceKind kind) {
    switch (kind) {
      case DeviceKind.V:
        addElement(IndependentSource(circuit: this, kind: DeviceKind.V));
        break;
      case DeviceKind.I:
        addElement(IndependentSource(circuit: this, kind: DeviceKind.I));
        break;
      case DeviceKind.R:
        addElement(Resistor(this));
        break;
      default:
        throw ArgumentError("Invalid element kind: $kind");
    }
  }

  void removeDevice(Device element) => elements.remove(element);
  void addNode(Node node) => nodes.add(node);
  void removeNode(Node node) => nodes.remove(node);
  int numNodes() => nodes.length;
  int numElements() => elements.length;

  void connect(Terminal terminal, Node node) {
    terminal.node = node;
    node.addTerminal(terminal);
  }
}
