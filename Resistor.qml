import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3
import QtQuick.Shapes 1.15
import com.company.models 1.0

Shape {
    layer.enabled: true
    layer.samples: 4
    property int increment: 10
    width: 9*increment
    height: 2*increment
    anchors.centerIn: parent

    ShapePath {
        strokeWidth: 2
        strokeColor: "black"
        strokeStyle: ShapePath.SolidLine
        startX: 0; startY: increment
        PathLine { x: 1.5*increment; y: increment }
        PathLine { x: 2*increment; y: 0 }
        PathLine { x: 3*increment; y: 2*increment }
        PathLine { x: 4*increment; y: 0 }
        PathLine { x: 5*increment; y: 2*increment }
        PathLine { x: 6*increment; y: 0 }
        PathLine { x: 7*increment; y: 2*increment }
        PathLine { x: 7.5*increment; y: increment }
        PathLine { x: 9*increment; y: increment }
    }
}
