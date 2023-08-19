import 'package:app/widgets/general/themes.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/widgets/diagram/diagram.dart';
import 'package:app/widgets/diagram/diagram_controls.dart';

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
      theme: lightTheme,
      title: 'Side Circuit',
      home: Scaffold(
        appBar: AppBar(
          title: Text('Side Circuit'),
          actions: [DiagramControls()],
        ),
        body: Diagram(),
      ),
    );
  }
}
