import 'package:flutter/material.dart';

import 'package:app/models/visual/wire_segment.dart';
import 'package:app/widgets/general/line.dart';

class WireSegmentView extends StatelessWidget {
  final WireSegment wireSegment;

  WireSegmentView({super.key, required this.wireSegment});

  @override
  Widget build(BuildContext context) {
    Offset start = wireSegment.start();
    Offset end = wireSegment.end();
    print('WireSegmentView: start=$start, end=$end');
    return Positioned(
      left: 0,
      top: 0,
      child: Line(
        start: start,
        end: end,
      ),
    );
  }
}
