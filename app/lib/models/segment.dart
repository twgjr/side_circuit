import 'package:app/models/diagram_symbol.dart';
import 'package:app/models/vertex.dart';
import 'package:app/models/wire.dart';

class Segment {
  final Wire wire;
  final Vertex start;
  final Vertex end;
  final DiagramSymbol symbol;

  Segment({required this.wire, required this.start, required this.end})
      : symbol = DiagramSymbol.segment(end: end.position());
}
