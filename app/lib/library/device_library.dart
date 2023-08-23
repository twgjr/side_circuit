import 'package:app/models/circuit.dart';
import 'package:app/models/device.dart';
import 'package:app/models/terminal.dart';
import 'package:flutter/material.dart';

class IndependentSource extends Device {
  IndependentSource(Circuit circuit, DeviceKind kind) : super(circuit, kind) {
    shape.reset();
    shape.addRect(100, 100);
    shape.end_path();
  }
}

class Resistor extends Device {
  Resistor(Circuit circuit) : super(circuit, DeviceKind.R) {
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
    t0.addPosition(offset - t0.shape.center());
    t1.addPosition(offset - t1.shape.center());
    terminals.add(t0);
    terminals.add(t1);
  }
}

class Block extends Device {
  Block(Circuit circuit) : super(circuit, DeviceKind.BLOCK) {
    shape.reset();
    shape.addRect(100, 100);
    shape.end_path();
  }
}
