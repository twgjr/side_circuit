import 'package:app/models/circuit/node.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/circuit/wire_segment.dart';
import 'package:app/models/view/vertex.dart';

class Wire {
  Terminal? terminal;
  Node? node;
  // Visual visual = Visual();
  List<WireSegment> segments = [];
  List<Vertex> vertices = [];

  Wire({this.terminal, this.node});

  Wire copyWith({Terminal? terminal, Node? node}) {
    final wire =
        Wire(terminal: terminal ?? this.terminal, node: node ?? this.node);
    for (WireSegment segment in segments) {
      wire.segments.add(WireSegment(
          wire, segment.length, segment.start.copy(), segment.end.copy()));
    }
    for (Vertex vertex in vertices) {
      wire.vertices.add(vertex.copy());
    }
    return wire;
  }

  void removeTerminal(Terminal terminal) {
    if (this.terminal == terminal) {
      this.terminal = null;
    }
  }
}
