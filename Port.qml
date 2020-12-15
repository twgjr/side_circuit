import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtQuick.Dialogs 1.2
import com.company.models 1.0

Rectangle {
    width: 10
    height: width
    radius: width/2
    border.color: "black"
    border.width: 2

    MouseArea {
        anchors.fill: parent
        drag.target: parent
        drag.threshold: 0
    }
}
