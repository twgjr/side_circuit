import 'package:app/models/visual/diagram_symbol.dart';
import 'package:app/models/visual/vertex.dart';

class Segment {
  final Vertex start;
  final Vertex end;
  final DiagramSymbol _symbol = DiagramSymbol();

  Segment({required this.start, required this.end}) {
    _symbol.shape.addCircle(10);
  }
}