import 'dart:math';

import 'port.dart';

class Link {
    Port? start;
    Port? end;
    Point? lastPoint;

    Link();

    void setEndPort(Port endPort)
    {
        if(this.end == endPort){
            return;
        }
        this.end = endPort;
        this.end!.connectedLinks.add(this);
    }

    void disconnectEndPort()
    {
        if(this.end!=null){
            this.end!.connectedLinks.remove(this);
        }
    }

    bool portConnected()
    {
        return this.end!=null?true:false;
    }
}
