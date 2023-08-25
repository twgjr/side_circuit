import 'package:app/models/vertex.dart';
import 'package:app/widgets/symbol/symbol_widget.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class VertexWidget extends ConsumerWidget {
  final Vertex vertex;

  VertexWidget({required this.vertex, super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: vertex.position().dx,
      top: vertex.position().dy,
      child: SymbolWidget(vertex.symbol),
    );
  }
}
