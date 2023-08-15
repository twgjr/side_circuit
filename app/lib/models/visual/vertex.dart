import 'package:app/models/visual/symbol.dart';

class Vertex {
  Symbol symbol = Symbol();

  Vertex();

  Vertex copy() {
    return Vertex()..symbol = symbol.copy();
  }
}
