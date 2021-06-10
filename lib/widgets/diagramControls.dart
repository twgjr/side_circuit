import 'package:flutter/material.dart';

import '../models/diagram.dart';

class DiagramControls extends StatelessWidget {
  final Diagram mainDiagram;
  final Function addDiagramItem;

  DiagramControls(this.mainDiagram, this.addDiagramItem);

  @override
  Widget build(BuildContext context) {
    return IconButton(
        icon: Icon(Icons.add),
        tooltip: "add new element",
        onPressed: () {
          addDiagramItem();
        });
  }
}
