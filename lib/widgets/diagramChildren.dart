import 'package:flutter/material.dart';

import '../models/diagram.dart';
import 'diagramChildItem.dart';

class DiagramChildren extends StatelessWidget {
  final Diagram mainDiagram;

  DiagramChildren(this.mainDiagram);

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
          border: Border.all(color: Colors.blueAccent)
      ),
      height: 300,
        width: 300,
        child: Stack(
      children: mainDiagram.proxyRoot.children.map((dItem) {
        return DiagramChildItem(dItem);
      }).toList(),
    ));
  }
}
