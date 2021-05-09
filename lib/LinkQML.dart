import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import QtQuick.Shapes 1.15
import com.company.models 1.0

Item{
    id: linkItemRoot
    height:implicitHeight
    width:implicitWidth

    Shape {
        id:shapeRoot
        ShapePath {
            id:shapeRootPath
            strokeWidth: 2
            strokeColor: "red"
            strokeStyle: ShapePath.DashLine
            dashPattern: [ 1, 5 ]
            startX: portId.width/2 ; startY: portId.height/2
            PathLine {
                id: segment
                x: endRect.x+endRect.radius
                y: endRect.y+endRect.radius
            }
        }
    }//Shape

    Rectangle{
        id: endRect

        property var endPortCenter: model.lastPoint
        property real endPortCenterX: endPortCenter.x
        property real endPortCenterY: endPortCenter.y
        property real baseX: endPortCenterX-itemRectId.x-portId.x-portId.radius
        property real baseY: endPortCenterY-itemRectId.y-portId.y-portId.radius
        property real deltaX: 0
        property real deltaY: 0
        property real correctedX: baseX+deltaX
        property real correctedY: baseY+deltaY

        Connections{
            target: portId
            function onPortAbsPositionChanged() {
                endRect.updateLinkPosition()
                endRect.x= endRect.correctedX
                endRect.y= endRect.correctedY
            }
        }

        function updateLinkPosition(){
            console.log("endPortCenter: "+endRect.endPortCenter)
            var itemAngle = itemRectId.rotation/180*Math.PI
            var itemCenter = [itemRectId.width/2,
                             itemRectId.height/2]
            var endPortCenter = [
                        endPortCenterX-itemRectId.x-itemCenter[0],
                        endPortCenterY-itemRectId.y-itemCenter[1]]
            var endPortDist = Math.sqrt(
                        Math.pow(Math.abs(endPortCenter[0]),2)+
                        Math.pow(Math.abs(endPortCenter[1]),2))
            var endPortBaseAngle = Math.atan2(
                        endPortCenter[0],
                        -endPortCenter[1])
            var endPortRotatedCenter = [
                        endPortDist*Math.sin(endPortBaseAngle-itemAngle),
                        -endPortDist*Math.cos(endPortBaseAngle-itemAngle)]
            var endPortRotatedBase = [
                        endPortRotatedCenter[0]+portId.x,
                        endPortRotatedCenter[1]+portId.y]
            var startPortCenter = [portId.x+portId.width/2,
                    portId.y+portId.height/2]
            endRect.deltaX = endPortRotatedCenter[0]-endPortCenter[0]
            endRect.deltaY = endPortRotatedCenter[1]-endPortCenter[1]
        }

        width: 10
        height: width
        radius: width/2
        border.color: "black"
        color: "transparent"

        Drag.active: linkMouseArea.drag.active
        Drag.keys: [ "dropKey" ]
        Drag.hotSpot.x: width/2
        Drag.hotSpot.y: height/2

        states: [
            State {
                name: "connected"; when: model.portConnected
                PropertyChanges {
                    target: linkMouseArea
                    drag.target: undefined
                    x: endRect.correctedX
                    y: endRect.correctedY
                }
                PropertyChanges {
                    target: shapeRootPath
                    strokeColor: "black"
                    strokeStyle: ShapePath.SolidLine
                }
            }
        ]

        Component.onCompleted:{
            if(model.portConnected){
                endRect.updateLinkPosition()
                endRect.x= endRect.correctedX
                endRect.y= endRect.correctedY
            } else {
                var extraX = 0
                var extraY = 0
                var extra = 10
                switch(sideNum){
                case 0: extraY = -extra; break;
                case 1: extraY = extra; break;
                case 2: extraX = -extra; break;
                case 3: extraX = extra; break;
                }
                x = extraX
                y = extraY
            }
        }

        MouseArea {
            id: linkMouseArea
            acceptedButtons: Qt.LeftButton | Qt.RightButton
            anchors.fill: parent
            drag.target: parent
            drag.threshold: 0
            drag.smoothed: false

            onPositionChanged: {
//                if(!model.portConnected){
//                    var linkAbsX = endRect.baseX
//                    var linkAbsY = endRect.baseY
//                    model.lastPoint = Qt.point(linkAbsX,linkAbsY)
//                }
            }

            onReleased: {
                //drop handler
                if(endRect.Drag.target !== null){
                    console.log("drag target: "+endRect.Drag.target)
                    dataSource.endLinkFromLink(model.thisLink)
                } else {
                    console.log("drag target was null")
                }
                parent.Drag.drop()
            }

            onClicked: {
                if(mouse.button & Qt.RightButton){
                    portContextMenu.popup()

                }
                if(mouse.button & Qt.LeftButton){
                    //add new segment and set new segment to follow mouse pointer
                }
            }

            onPressed: {
                endRect.state = ""
                dataSource.disconnectPortfromLink(model.thisLink)
            }


        }//MouseArea
    }//Rectangle

    Menu {
        id: portContextMenu
        MenuItem {
            text: "Delete Link"
            onTriggered: {
                dataSource.deleteLink(parentIndex,proxyPortIndex,model.index)
            }
        }
    } //Menu
}//Item
