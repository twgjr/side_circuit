import 'dart:math';

import 'diagram.dart';
import 'link.dart';

class Port {

    DiagramItem? itemParent;
    List<Link> links = [];
    List<Link> connectedLinks = [];

    String name = "";
    Point? absPoint;
    bool linkIsValid = false;

    Port();

    void startLink()
    {
        Link newLink = Link();
        newLink.start = this;
        links.add(newLink);
    }
}
