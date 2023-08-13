import 'package:app/models/circuit/wire.dart';
import 'package:app/models/view/vertex.dart';

class WireSegment {
  final Wire wire;
  final double length;
  final Vertex start;
  final Vertex end;

  WireSegment(this.wire, this.length, this.start, this.end);
}
