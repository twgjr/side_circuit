import 'package:flutter/material.dart';

import '../../models/diagram.dart';
import 'level/diagram_level.dart';

class DiagramArea extends StatefulWidget {
  final void Function(DiagramItem) downLevel;
  final void Function(DiagramItem) deleteItem;
  final List<DiagramItem> dItems;
  final int depth;

  DiagramArea({
    super.key,
    required this.downLevel,
    required this.deleteItem,
    required this.depth,
    required this.dItems,
  });

  @override
  _DiagramAreaState createState() => _DiagramAreaState();
}

class _DiagramAreaState extends State<DiagramArea> {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            Text("depth: ${widget.depth}"),
          ],
        ),
        DiagramLevel(
          dItems: widget.dItems,
          downLevel: widget.downLevel,
          deleteItem: widget.deleteItem,
        ),
      ],
    );
  }
}
