class Vertex {
  double x;
  double y;
  double angleIncrement;

  Vertex({required this.x, required this.y, required this.angleIncrement});

  Vertex copy() {
    return Vertex(x: x, y: y, angleIncrement: angleIncrement);
  }
}
