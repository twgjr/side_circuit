import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/node.dart';
import 'package:app/providers/circuit_provider.dart';

class NodeView extends ConsumerWidget {
  final Node node;

  NodeView({required this.node});

  void _showPopupMenu(Offset position, BuildContext context, WidgetRef ref) {
    final RenderBox overlay =
        Overlay.of(context).context.findRenderObject() as RenderBox;

    final relativePosition = RelativeRect.fromSize(
      Rect.fromLTWH(position.dx, position.dy, 0, 0),
      overlay.size,
    );

    showMenu(
      context: context,
      position: relativePosition,
      items: [
        PopupMenuItem(value: "delete", child: Text("Delete")),
      ],
    ).then((value) {
      if (value != null) {
        switch (value) {
          case "delete":
            ref.read(circuitProvider.notifier).removeNode(node);
            break;
        }
      }
    });
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: node.symbol.position.dx,
      top: node.symbol.position.dy,
      child: GestureDetector(
        onPanUpdate: (details) {
          node.symbol.position += details.delta;
        },
        onSecondaryTapDown: (details) {
          _showPopupMenu(details.globalPosition, context, ref);
        },
        child: Card(
          child: Padding(
            padding: const EdgeInsets.all(8.0),
            child: Column(
              children: [
                Text('Node ${node.id}'),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
