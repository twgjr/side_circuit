import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'widgets/diagram/diagram.dart';
import 'widgets/diagram/diagram_controls.dart';

void main() => runApp(
      ProviderScope(
        child: MainApp(),
      ),
    );

class MainApp extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => _MainApp();
}

class _MainApp extends State<MainApp> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true),
      title: 'Side Circuit',
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: Theme.of(context).colorScheme.primary,
          title: Text('Side Circuit'),
          actions: [DiagramControls()],
        ),
        body: Diagram(),
      ),
    );
  }
}
