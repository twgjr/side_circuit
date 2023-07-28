import 'package:app/models/circuit/device.dart';
import 'package:flutter/material.dart';

class DeviceView extends StatelessWidget {
  final Device device;
  final BuildContext cktViewCtx;

  DeviceView({
    required this.device,
    required this.cktViewCtx,
  });

  void _showPopupMenu(Offset position, BuildContext context) {
    final RenderBox overlay =
        Overlay.of(cktViewCtx).context.findRenderObject() as RenderBox;

    final relativePosition = RelativeRect.fromSize(
      Rect.fromLTWH(position.dx, position.dy, 0, 0),
      overlay.size,
    );

    showMenu(
      context: context,
      position: relativePosition,
      items: [
        PopupMenuItem(value: "edit", child: Text("Edit")),
        PopupMenuItem(value: "delete", child: Text("Delete")),
      ],
    ).then((value) {
      if (value != null) {
        switch (value) {
          case "delete":
            print('not implemented yet');
            break;
          case "edit":
            print("Not implemented yet");
            break;
        }
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onSecondaryTapDown: (details) {
        _showPopupMenu(details.globalPosition, context);
      },
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(8.0),
          child: Column(
            children: [
              Text('${device.kind.name}${device.id}'),
            ],
          ),
        ),
      ),
    );
  }
}
