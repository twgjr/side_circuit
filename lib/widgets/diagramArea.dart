import 'package:flutter/material.dart';
import 'package:side_circuit/main.dart';

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
      _mainDiagram.proxyRoot.addChild();
    });
  }

  void _downLevel(DiagramItem dItem) {
    setState(() {
      _mainDiagram.moveDown(dItem);
    });
  }

  void _upLevel() {
    setState(() {
      _mainDiagram.moveUp();
    });
  }

  void _topLevel() {
    setState(() {
      _mainDiagram.moveToTop();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(children: [
      DiagramControls(_addNewDiagramItem, _upLevel, _topLevel),
      Row(
        children: [
          Text("depth: ${_mainDiagram.proxyRoot.depth()}"),
          Text(" breadth: ${_mainDiagram.proxyRoot.breadth()}"),
        ],
      ),
      Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.max,
        children: [
          DiagramChildren(_mainDiagram.proxyRoot.children, _downLevel,context),
        ],
      ),
    ]);
  }
}
