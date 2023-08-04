import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'widgets/diagram/diagram.dart';
import 'widgets/diagram/diagram_controls.dart';
import 'models/view/visual.dart';

void main() => runApp(
      ProviderScope(
        child: MaterialApp(
          debugShowCheckedModeBanner: false,
          title: 'Side Circuit',
          home: MainApp(),
        ),
      ),
    );

class MainApp extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => _MainApp();
}

class _MainApp extends State<MainApp> {
  List<Visual> dataObjs = [];

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
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
