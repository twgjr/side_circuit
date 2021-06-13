import 'package:flutter/material.dart';

import '../models/diagram.dart';
import 'diagramArea.dart';
import 'diagramChildItem.dart';

class DiagramChildren extends StatelessWidget {
  final List<DiagramItem> _dItems;
  final DiagramAreaState diagramAreaState;
  final BuildContext _diagramAreaContext;

  DiagramChildren(this._dItems, this.diagramAreaState, this._diagramAreaContext);

  @override
  Widget build(BuildContext context) {
    return Container(
        decoration: BoxDecoration(border: Border.all(color: Colors.blueAccent)),
        height: 300,
        child: Stack(
          children: _dItems.map((dItem) {
            /*
              Using unique key to force the children to completely
              redraw.  Fix error when switching proxyRoot.
             */
            return DiagramChildItem(UniqueKey(),dItem, diagramAreaState,_diagramAreaContext);
          }).toList(),
        ));
  }
}
