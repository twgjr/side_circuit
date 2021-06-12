import 'package:flutter/material.dart';

import '../models/diagram.dart';
import 'diagramChildItem.dart';

class DiagramChildren extends StatelessWidget {
  final List<DiagramItem> _dItems;
  final Function _downLevel;

  DiagramChildren(this._dItems, this._downLevel);

  @override
  Widget build(BuildContext context) {
    return Container(
        decoration: BoxDecoration(border: Border.all(color: Colors.blueAccent)),
        height: 300,
        //width: 300,
        child: Stack(
          children: _dItems.map((dItem) {
            return DiagramChildItem(dItem, _downLevel);
          }).toList(),
        ));
  }
}
