import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/widgets/diagram/diagram.dart';
import 'package:app/widgets/diagram/diagram_controls.dart';
import 'package:app/hotkeys.dart';

void main() => runApp(
      ProviderScope(
        child: MainApp(),
      ),
    );

class MainApp extends ConsumerWidget {
  MainApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Side Circuit',
      home: HotKeys(
        child: Scaffold(
          appBar: AppBar(
            title: Text('Side Circuit'),
            actions: [DiagramControls()],
          ),
          body: Diagram(),
        ),
      ),
    );
  }
}
