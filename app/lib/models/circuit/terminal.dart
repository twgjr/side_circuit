import 'package:app/models/visual/vector_path.dart';
import 'package:flutter/material.dart';

import 'package:app/models/circuit/device.dart';
import 'package:app/models/visual/wire.dart';
import 'package:app/models/visual/visual.dart';

class Terminal {
  Device device;
  String name;
  Wire? wire;
  Visual visual = Visual();

  Terminal(this.device, this.name) {
    visual.shape = VectorPath();
    visual.shape.addCircle(5);
  }

  Terminal copyWith({Device? device, Wire? wire}) {
    final terminal = Terminal(device ?? this.device, name);
    terminal.wire = wire ?? this.wire;
    terminal.visual = visual.copy();
    return terminal;
  }

  Offset get position => visual.position;
  Offset globalPosition() => device.visual.position + position;
  Offset center() => visual.center();
  Offset globalCenter() => device.visual.position + position + center();
}
