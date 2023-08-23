import 'package:app/models/terminal.dart';
import 'package:app/models/node.dart';
import 'package:app/models/wire.dart';
import 'package:flutter/material.dart';

class Net {
  List<Wire> wires = [];
  List<Node> nodes = [];

  Net();

  Wire startWire({Terminal? terminal, Offset? position}) {
    Wire wire = Wire();
    wire.start(terminal: terminal, position: position);
    wires.add(wire);
    return wire;
  }
}
