import 'package:flutter/material.dart';

import 'package:app/models/circuit/device.dart';
import 'package:app/models/visual/wire.dart';
import 'package:app/models/visual/symbol.dart';

class Terminal {
  Device device;
  String name;
  Wire? wire;
  Symbol symbol = Symbol();

  Terminal(this.device, this.name) {
    symbol.shape.addRect(5, 5);
  }

  Terminal copyWith({Device? device, Wire? wire}) {
    final terminal = Terminal(device ?? this.device, name);
    terminal.wire = wire ?? this.wire;
    terminal.symbol = symbol.copy();
    return terminal;
  }

  Offset get position => symbol.position;
  Offset globalPosition() => device.symbol.position + position;
  Offset center() => symbol.center();
  Offset globalCenter() => device.symbol.position + position + center();
}
