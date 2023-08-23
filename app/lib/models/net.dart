import 'package:app/models/node.dart';
import 'package:app/models/wire.dart';

class Net {
  List<Wire> wires = [];
  List<Node> nodes = [];

  Net();

  void addWire(Wire wire) {
    wires.add(wire);
  }
}
