import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/wire_segment.dart';

class WireSegmentView extends ConsumerWidget {
  final WireSegment wireSegment;

  WireSegmentView({required this.wireSegment, super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container();
  }
}
