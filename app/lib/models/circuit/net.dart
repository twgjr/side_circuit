import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/visual/node.dart';
import 'package:app/models/visual/vertex.dart';
import 'package:app/models/visual/wire.dart';

class Net {
  List<Wire> wires = [];
  List<Node> nodes = [];

  Net();

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

  // Wire startWire({Terminal? terminal, Offset? position}) {
  //   final wire = Wire();
  //   wires.add(wire);
  //   wire.start(terminal: terminal, position: position);
  //   return wire;
  // }
}
