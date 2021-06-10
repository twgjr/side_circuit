import 'package:flutter/material.dart';

import '../models/diagram.dart';

class DiagramChildItem extends StatelessWidget {
  final DiagramItem dItem;

  DiagramChildItem(this.dItem);

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 50,
      left: 50,
      child: Draggable(
        child: Container(
          child: Text(dItem.equations[0].equationString),
        ),
        feedback: Text("dragging"),
      ),
    );
  }
}
