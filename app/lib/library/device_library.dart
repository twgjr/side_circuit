import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:flutter/material.dart';

class IndependentSource extends Device {
  IndependentSource({
    required Circuit circuit,
    required DeviceKind kind,
  }) : super(circuit: circuit, kind: kind) {
    shape.reset();
    shape.addRect(100, 100);
    shape.end_path();
  }
}

class Resistor extends Device {
  Resistor(Circuit circuit) : super(circuit: circuit, kind: DeviceKind.R) {
    shape.reset();
    Terminal t0 = Terminal(this);
    t0.setPosition(shape.currentPoint);
    shape.angleLine(0, 10);
    shape.angleLine(-60, 10);
    shape.angleLine(60, 20);
    shape.angleLine(-60, 20);
    shape.angleLine(60, 20);
    shape.angleLine(-60, 20);
    shape.angleLine(60, 20);
    shape.angleLine(-60, 10);
    shape.angleLine(0, 10);
    Terminal t1 = Terminal(this);
    t1.setPosition(shape.currentPoint);
    Offset offset = shape.end_path();
    t0.updatePosition(offset - t0.shape.center());
    t1.updatePosition(offset - t1.shape.center());
    terminals.add(t0);
    terminals.add(t1);
  }
}

class Block extends Device {
  Block(Circuit circuit) : super(circuit: circuit, kind: DeviceKind.R) {
    shape.reset();
    shape.addRect(100, 100);
    shape.end_path();
  }
}
