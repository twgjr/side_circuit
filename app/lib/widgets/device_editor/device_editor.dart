import 'package:app/widgets/device_editor/device_editor_area.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/device_providers.dart';
import 'package:app/widgets/device_editor/device_editor_top_bar.dart';
import 'package:app/widgets/device_editor/device_editor_toolbar.dart';

class DeviceEditor extends ConsumerStatefulWidget {
  DeviceEditor({super.key});

  @override
  DeviceEditorState createState() => DeviceEditorState();
}

class DeviceEditorState extends ConsumerState<DeviceEditor> {
  Offset position = Offset(0, 0);
  Size size = Size(400, 400);

  Offset getCenterOffset(Size size) {
    return Offset(size.width / 2, size.height / 2);
  }

  void addOffset(Offset position) {
    setState(() {
      this.position += position;
    });
  }

  @override
  void initState() {
    final RenderBox diagramRenderBox =
        Overlay.of(context).context.findRenderObject() as RenderBox;
    size = diagramRenderBox.size / 2;
    position = Offset(size.width / 2, size.height / 2);
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    final device = ref.watch(deviceChangeProvider);
    return Flex(
      direction: Axis.horizontal,
      children: [
        Expanded(
          child: Container(
            child: Stack(
              children: [
                Positioned(
                  left: position.dx,
                  top: position.dy,
                  child: Card(
                    clipBehavior: Clip.hardEdge,
                    elevation: 10,
                    shadowColor: Colors.black,
                    child: Container(
                      decoration: BoxDecoration(
                        border: Border.all(
                          color: Theme.of(context).primaryColorDark,
                          width: 1.0,
                        ),
                      ),
                      width: size.width,
                      height: size.height,
                      child: Column(
                        children: [
                          DeviceEditorTopBar(onDrag: addOffset),
                          DeviceEditorToolbar(),
                          Expanded(
                            child: Container(
                                decoration: BoxDecoration(
                                  border: Border.all(
                                    color: Colors.black,
                                    width: 2,
                                  ),
                                ),
                                child: DeviceEditorArea(device: device)),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
