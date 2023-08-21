import 'package:app/models/visual/vertex.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class VertexView extends ConsumerWidget {
  final Vertex vertex;

  VertexView({required this.vertex, super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: vertex.diagramPosition.dx - 2,
      top: vertex.diagramPosition.dy - 2,
      child: Container(
        width: 4,
        height: 4,
        decoration: BoxDecoration(
          color: Colors.black,
          shape: BoxShape.circle,
        ),
      ),
    );
  }
}
