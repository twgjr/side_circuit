import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import com.company.models 1.0

Rectangle{
    opacity: 0
    MouseArea{
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton
        hoverEnabled: true
        drag.target: parent
        onEntered: cursorShape = Qt.SizeFDiagCursor


        onPositionChanged: {
            if(pressed){
                blkRectId.height = Math.max(blkRectId.height+mouseY,blkRectId.border.width*4)
                blkRectId.width = Math.max(blkRectId.width+mouseX,blkRectId.border.width*4)
            }
        }
    }
}
