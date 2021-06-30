import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

import 'preventDefault.dart';
import 'widgets/diagramArea.dart';

void main(){
  if(kIsWeb) {
    PreventDefault rightClickMenu = PreventDefault();
  }
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
