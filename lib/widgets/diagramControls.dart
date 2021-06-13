import 'package:flutter/material.dart';

import 'diagramArea.dart';

class DiagramControls extends StatelessWidget {
  final DiagramAreaState diagramAreaState;

  DiagramControls(/*this._addDiagramItem,*/ this.diagramAreaState);

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        IconButton(
            icon: Icon(Icons.home),
            tooltip: "go to top level",
            onPressed: () {
              diagramAreaState.topLevel();
            }),
        IconButton(
            icon: Icon(Icons.arrow_back),
            tooltip: "go up one level",
            onPressed: () {
              diagramAreaState.upLevel();
            }),
        IconButton(
            icon: Icon(Icons.add),
            tooltip: "add new element",
            onPressed: () {
              diagramAreaState.addItem();
            }),
      ],
    );
  }
}
