import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/visual/node.dart';
import 'package:app/models/visual/vertex.dart';
import 'package:app/models/visual/wire.dart';
import 'package:flutter/material.dart';

class Net {
  Circuit circuit;
  List<Wire> wires = [];
  List<Node> nodes = [];

  Net(this.circuit);

  // List<Terminal> terminals() {
  //   List<Terminal> terminals = [];
  //   for (Wire wire in wires) {
  //     Vertex start = wire.start();
  //     if (start.terminal != null) {
  //       terminals.add(start.terminal!);
  //     }
  //     Vertex end = wire.end();
  //     if (end.terminal != null) {
  //       terminals.add(end.terminal!);
  //     }
  //   }
  //   return terminals;
  // }

  Vertex startWire({Terminal? terminal, required Offset position}) {
    Wire wire = Wire(this);
    wire.start(terminal: terminal, position: position);
    wires.add(wire);
    return wire.last();
  }
}
