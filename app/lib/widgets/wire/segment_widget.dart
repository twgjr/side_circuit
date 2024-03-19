import 'package:app/providers/circuit_provider.dart';
import 'package:app/providers/mode_provider.dart';
import 'package:app/widgets/symbol/shape_painter.dart';
import 'package:app/widgets/symbol/shape_widget.dart';
import 'package:flutter/material.dart';

import 'package:app/models/segment.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class SegmentWidget extends ConsumerWidget {
  final Segment segment;

  SegmentWidget({super.key, required this.segment});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final modeWatch = ref.watch(modeProvider);
    final circuitRead = ref.read(circuitProvider.notifier);
    Offset start = segment.start.position();
    Offset end = segment.end.position();
    return Positioned(
      left: start.dx,
      top: start.dy,
      child: GestureDetector(
        onTap: () {
          if (modeWatch.delete) {
            circuitRead.removeWireContaining(segment);
            print('delete segment');
          }
        },
        child: ShapeWidget(
          shape: ShapePainter.segment(end: end - start),
        ),
      ),
    );
  }
}
