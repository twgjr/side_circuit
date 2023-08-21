import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/node.dart';

class NodeView extends ConsumerWidget {
  final Node node;

  NodeView({required this.node});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: node.position().dx,
      top: node.position().dy,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(8.0),
          child: Column(
            children: [
              Text('Node'),
            ],
          ),
        ),
      ),
    );
  }
}
