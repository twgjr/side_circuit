import 'package:app/models/circuit.dart';
import 'package:app/models/terminal.dart';
import 'package:app/models/node.dart';
import 'package:app/models/vertex.dart';
import 'package:app/models/wire.dart';
import 'package:flutter/material.dart';

class Net {
  Circuit circuit;
  List<Wire> wires = [];
  List<Node> nodes = [];

  Net(this.circuit);

  Vertex startWire({Terminal? terminal, required Offset position}) {
    Wire wire = Wire(this);
    wire.start(terminal: terminal, position: position);
    wires.add(wire);
    return wire.last();
  }
}
