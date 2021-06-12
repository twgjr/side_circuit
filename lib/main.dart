import 'package:flutter/material.dart';
import 'dart:html';

import 'widgets/diagramArea.dart';

void main(){
  window.document.onContextMenu.listen((evt) => evt.preventDefault());
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Side Circuit',
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatelessWidget {

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Side Circuit'),
      ),
      body: DiagramArea(),
    );
  }
}
