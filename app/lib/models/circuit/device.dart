import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/terminal.dart';

enum DeviceKind { V, I, R, VC, CC, SW, L, C, VG, CG }

abstract class Device {
  Circuit circuit;
  final List<Terminal> terminals = [];
  DeviceKind? kind;

  Device({required this.circuit, this.kind});

  int index() => circuit.elements.indexOf(this);
}

class IndependentSource extends Device {
  IndependentSource({
    required Circuit circuit,
    required DeviceKind kind,
  }) : super(circuit: circuit, kind: kind);
}

class Resistor extends Device {
  Resistor(Circuit circuit) : super(circuit: circuit, kind: DeviceKind.R);
}
