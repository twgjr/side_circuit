import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';

class IndependentSource extends Device {
  IndependentSource({
    required Circuit circuit,
    required DeviceKind kind,
  }) : super(circuit: circuit, kind: kind) {
    visual.shape.reset();
    visual.shape.addRect(100, 100);
    visual.shape.end_path();
  }
}

class Resistor extends Device {
  Resistor(Circuit circuit) : super(circuit: circuit, kind: DeviceKind.R) {
    final shape = visual.shape;
    shape.reset();
    shape.angleLine(0, 10);
    shape.angleLine(-60, 10);
    shape.angleLine(60, 20);
    shape.angleLine(-60, 20);
    shape.angleLine(60, 20);
    shape.angleLine(-60, 20);
    shape.angleLine(60, 20);
    shape.angleLine(-60, 10);
    shape.angleLine(0, 10);
    shape.end_path();
  }
}

class Block extends Device {
  Block(Circuit circuit) : super(circuit: circuit, kind: DeviceKind.R) {
    visual.shape.reset();
    visual.shape.addRect(100, 100);
    visual.shape.end_path();
  }
}
