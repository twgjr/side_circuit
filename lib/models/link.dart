import 'dart:math';

import 'port.dart';

class Link {
    Port start;
    Port end;
    Point lastPoint;

    Link();

    void setEndPort(Port endPort)
    {
        if(end == endPort){
            return;
        }
        end = endPort;
        end.connectedLinks.add(this);
    }

    void disconnectEndPort()
    {
        if(end!=null){
            end.connectedLinks.remove(this);
        }
    }

    bool portConnected()
    {
        return end!=null?true:false;
    }
}
