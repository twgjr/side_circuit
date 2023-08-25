import 'package:app/models/circuit.dart';
import 'package:app/models/device.dart';
import 'package:app/models/terminal.dart';
import 'package:flutter/material.dart';

class IndependentSource extends Device {
  IndependentSource(Circuit circuit, DeviceKind kind) : super(circuit, kind) {
    symbol.shape.reset();
    symbol.shape.addRect(100, 100);
    symbol.shape.shiftPath();
  }
}

class Resistor extends Device {
  Resistor(Circuit circuit) : super(circuit, DeviceKind.R) {
    symbol.shape.reset();
    Terminal t0 = Terminal(this);
    t0.setPosition(symbol.shape.currentPoint);
    symbol.shape.angleLine(0, 10);
    symbol.shape.angleLine(-60, 10);
    symbol.shape.angleLine(60, 20);
    symbol.shape.angleLine(-60, 20);
    symbol.shape.angleLine(60, 20);
    symbol.shape.angleLine(-60, 20);
    symbol.shape.angleLine(60, 20);
    symbol.shape.angleLine(-60, 10);
    symbol.shape.angleLine(0, 10);
    Terminal t1 = Terminal(this);
    t1.setPosition(symbol.shape.currentPoint);
    Offset offset = symbol.shape.shiftPath();
    t0.addPosition(t0.symbol.center());
    t1.addPosition(t1.symbol.center());
    t0.addPosition(offset);
    t1.addPosition(offset);
    terminals.add(t0);
    terminals.add(t1);
  }
}

class Block extends Device {
  Block(Circuit circuit) : super(circuit, DeviceKind.BLOCK) {
    symbol.shape.reset();
    symbol.shape.addRect(100, 100);
    symbol.shape.shiftPath();
  }
}
