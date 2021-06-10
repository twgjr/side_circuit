import 'package:flutter/material.dart';

import '../models/diagram.dart';
import 'diagramChildren.dart';
import 'diagramControls.dart';

class DiagramArea extends StatefulWidget {
  @override
  _DiagramAreaState createState() => _DiagramAreaState();
}

class _DiagramAreaState extends State<DiagramArea> {
  final Diagram _mainDiagram = Diagram();

  void _addNewDiagramItem() {
    setState(() {
      _mainDiagram.proxyRoot.children.add(DiagramItem.child(_mainDiagram.proxyRoot));
    });
    print("added new child: ${_mainDiagram.proxyRoot.children.length}");
  }

  @override
  Widget build(BuildContext context) {
    return Column(children: [
      DiagramControls(_mainDiagram,_addNewDiagramItem),
      DiagramChildren(_mainDiagram),
    ]);
  }
}
