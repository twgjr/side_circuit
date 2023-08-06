import 'package:flutter/material.dart';

class PopupMenu extends StatelessWidget {
  final RelativeRect relativePosition;
  final List<MenuItemButton> children;

  PopupMenu({
    required this.relativePosition,
    required this.children,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: relativePosition.left,
      top: relativePosition.top,
      child: Card(
        clipBehavior: Clip.hardEdge,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(height: 8),
            ...children,
            SizedBox(height: 8),
          ],
        ),
      ),
    );
  }
}
