#! /usr/bin/env python

import sys
import rospy
import signal
from copy import deepcopy
from deformable_manipulation_msgs.srv import TransitionTestingVisualization
import std_msgs.msg

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QFont


class Widget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.data_folder = rospy.get_param("transition_learning_data_generation_node/data_folder", "/tmp")
        self.last_folder_used = deepcopy(self.data_folder)

        self.font = QFont("unexistent")
        self.font.setStyleHint(QFont.Monospace)

        self.createSourceFileBox()
        self.createTestFileBox()
        self.createTestFileListBox()
        self.createVisIdFeedbackBox()

        self.vis_id_to_item = {}

        self.layout = QGridLayout()
        self.layout.addWidget(self.source_file_box)
        self.layout.addWidget(self.test_file_box)
        self.layout.addWidget(self.test_file_list_box)
        self.layout.addWidget(self.vis_id_box)

        self.setLayout(self.layout)
        self.resize(1600, 700)
        self.move(2000, 100)

    def createSourceFileBox(self):
        self.source_file_box = QGroupBox("Source Transition")

        text_field = QLineEdit()
        text_field.setFont(self.font)
        set_button = QPushButton("Set")
        browse_button = QPushButton("Browse")

        layout = QGridLayout()
        layout.addWidget(text_field, 0, 0)
        layout.addWidget(set_button, 0, 1)
        layout.addWidget(browse_button, 0, 2)
        self.source_file_box.setLayout(layout)

        set_button.clicked.connect(lambda: self.setSourceTransition(text_field))
        browse_button.clicked.connect(lambda: self.fileBrowse(text_field, self.last_folder_used))

    def createTestFileBox(self):
        self.test_file_box = QGroupBox("Test Transition")

        text_field = QLineEdit()
        text_field.setFont(self.font)
        add_vis0_button = QPushButton("Add Id 0")
        add_button = QPushButton("Add")
        browse_button = QPushButton("Browse")

        layout = QGridLayout()
        layout.addWidget(text_field, 0, 0)
        layout.addWidget(add_vis0_button, 0, 1)
        layout.addWidget(add_button, 0, 2)
        layout.addWidget(browse_button, 0, 3)
        self.test_file_box.setLayout(layout)

        add_vis0_button.clicked.connect(lambda: self.addVisualizationVisId0(text_field=text_field))
        add_button.clicked.connect(lambda: self.addVisualization(text_field=text_field))
        browse_button.clicked.connect(lambda: self.fileBrowse(text_field, self.last_folder_used))

    def createTestFileListBox(self):
        self.test_file_list_box = QGroupBox("Test Transition List")

        text_field = QPlainTextEdit()
        text_field.setFont(self.font)
        # text_field.setAcceptRichText(False)
        # text_field.setFont
        add_vis0_button = QPushButton("Add Slowly Id 0")
        add_button = QPushButton("Add")
        clear_button = QPushButton("Clear")

        layout = QGridLayout()
        layout.addWidget(text_field, 0, 0, 1, 3)
        layout.addWidget(add_vis0_button, 1, 0)
        layout.addWidget(add_button, 1, 1)
        layout.addWidget(clear_button, 1, 2)
        self.test_file_list_box.setLayout(layout)

        add_vis0_button.clicked.connect(lambda: self.addMultipleFilesVisId0(text_field))
        add_button.clicked.connect(lambda: self.addMultipleFiles(text_field))
        clear_button.clicked.connect(lambda: text_field.setPlainText(""))

    def createVisIdFeedbackBox(self):
        self.vis_id_box = QGroupBox("Visualized Adapatations")

        self.ID_IDX = 0
        self.DELETE_IDX = 1
        self.FILENAME_IDX = 2

        self.vis_id_model = QStandardItemModel(0, 3)
        self.vis_id_model.setHorizontalHeaderLabels(["Vis ID", "Delete", "Filename"])

        self.vis_id_view = QTreeView(self)
        self.vis_id_view.setModel(self.vis_id_model)
        self.vis_id_view.setAlternatingRowColors(True)
        self.vis_id_view.setSortingEnabled(True)
        self.vis_id_view.setColumnWidth(self.ID_IDX, 50)
        self.vis_id_view.setColumnWidth(self.DELETE_IDX, 75)

        layout = QGridLayout()
        layout.addWidget(self.vis_id_view)
        self.vis_id_box.setLayout(layout)

    def fileBrowse(self, text_field, directory=None):
        file, filter = QFileDialog.getOpenFileName(parent=self, directory=directory, filter="*test_results.compressed")
        text_field.setText(file[len(self.data_folder) + 1:])

    def setSourceTransition(self, text_field):
        try:
            rospy.wait_for_service("transition_vis/set_source", timeout=1.0)
            set_source = rospy.ServiceProxy("transition_vis/set_source", TransitionTestingVisualization)
            set_source(text_field.text())
        except rospy.ROSException as e:
            print e
        except rospy.ServiceException as e:
            print "set_source service call failed: ", e

    def addVisualizationVisId0(self, filename=None, text_field=None):
        self.vis_id_pub = rospy.Publisher("transition_vis/set_next_vis_id", std_msgs.msg.Int32, queue_size=1, latch=True)
        self.vis_id_pub.publish(0)
        rospy.sleep(0.1)
        self.addVisualization(filename, text_field)

    def addVisualization(self, filename=None, text_field=None):
        if filename is None:
            filename = text_field.text()
        try:
            rospy.wait_for_service("transition_vis/add_visualization", timeout=1.0)
            add_visualization = rospy.ServiceProxy("transition_vis/add_visualization", TransitionTestingVisualization)
            id = add_visualization(filename).response

            if id in self.vis_id_to_item.keys():
                self.vis_id_model.removeRow(self.vis_id_to_item[id].row())

            id_item = QStandardItem(id)
            id_item.setFont(self.font)
            filename_item = QStandardItem(filename)
            filename_item.setFont(self.font)
            self.vis_id_model.appendRow([id_item, QStandardItem(), filename_item])

            delete_button = QPushButton()
            delete_button.setText("Delete")
            delete_button.clicked.connect(lambda: self.removeVisualization(id_item))
            delete_button_index = self.vis_id_model.index(id_item.row(), self.DELETE_IDX)
            self.vis_id_view.setIndexWidget(delete_button_index, delete_button)

            self.vis_id_to_item[id] = id_item

        except rospy.ROSException as e:
            print e
        except rospy.ServiceException, e:
            print "add_visualization service call failed: ", e

    def removeVisualization(self, id_item):
        try:
            rospy.wait_for_service("transition_vis/remove_visualization", timeout=1.0)
            remove_visualization = rospy.ServiceProxy("transition_vis/remove_visualization", TransitionTestingVisualization)
            remove_visualization(id_item.text())
        except rospy.ROSException as e:
            print e
        except rospy.ServiceException, e:
            print "remove_visualization service call failed: ", e
        finally:
            self.vis_id_to_item.pop(id_item.text())
            self.vis_id_model.removeRow(id_item.row())

    def parseFilename(self, text):
        text = str(text)
        text = text.strip()
        if text[0] == '"':
            text = text[1:]
        if text[-1] == '"':
            text = text[:-1]
        return text

    def addMultipleFilesVisId0(self, text_field):
        text = str(text_field.toPlainText())
        filelist = str.splitlines(text)
        filelist = [self.parseFilename(file) for file in filelist]
        text_field.setPlainText("\n".join(filelist))
        for file in filelist:
            self.addVisualizationVisId0(filename=file)
            raw_input("Press Enter to continue...")

    def addMultipleFiles(self, text_field):
        text = str(text_field.toPlainText())
        filelist = str.splitlines(text)
        filelist = [self.parseFilename(file) for file in filelist]
        text_field.setPlainText("\n".join(filelist))
        for file in filelist:
            self.addVisualization(filename=file)

if __name__ == "__main__":
    rospy.init_node("transition_file_browser", anonymous=True)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    widget = Widget()
    widget.setWindowTitle("Data Visualization")
    widget.show()

    sys.exit(app.exec_())