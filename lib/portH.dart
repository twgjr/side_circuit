import 'dart:math';

import 'diagramitemH.dart';
import 'linkH.dart';

class Port {

    DiagramItem itemParent;
    List<Link> links;
    List<Link> connectedLinks;

    String name;
    Point absPoint;
    bool linkIsValid;

    Port();

    void startLink()
    {
        Link newLink = Link();
        newLink.start = this;
        links.add(newLink);
    }
}
