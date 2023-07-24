import 'package:flutter/material.dart';

import '../../../models/diagram.dart';
import 'diagram_level_item.dart';

class DiagramLevel extends StatelessWidget {
  final List<DiagramItem> dItems;
  final void Function(DiagramItem) downLevel;
  final void Function(DiagramItem) deleteItem;

  DiagramLevel({
    super.key,
    required this.dItems,
    required this.downLevel,
    required this.deleteItem,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Stack(
        children: dItems.map(
          (dItem) {
            return DiagramChildItem(
              dItem: dItem,
              downLevel: downLevel,
              deleteItem: deleteItem,
              diagramAreaContext: context,
            );
          },
        ).toList(),
      ),
    );
  }
}
