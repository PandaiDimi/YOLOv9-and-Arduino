# We initialize the necessary PyQt5 libraries and some initial variables.
# These variables are used later in the application.
import sys
import os
from PyQt5.QtWidgets import QToolButton, QStyle, QApplication, QSpacerItem, QSizePolicy, QMainWindow, QPushButton, QComboBox, QVBoxLayout, QWidget, QMessageBox, QLineEdit, QFileDialog, QScrollArea, QHBoxLayout, QCheckBox, QLabel, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap

x = -1
y = 0
text = 0
text_dataset = 0
text_weight = 0
thress = 0
max_det = 1
save = ""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # We define the title and the minimum size of the window.
        self.setWindowTitle("My App")
        self.setMinimumSize(1100, 1000)

        # We change the current directory to the directory of the script.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        # We create horizontal layouts for the layout of the widgets inside the window.
        # Horizontal layout for source save and file selection.
        h_layout1 = QHBoxLayout()
        h_layout2 = QHBoxLayout()
        h_layout3 = QHBoxLayout()
        h_layout4 = QVBoxLayout()
        h_layout5 = QVBoxLayout()
        h_layout6 = QHBoxLayout()
        h_layout7 = QHBoxLayout()
        h_layout8 = QHBoxLayout()
        h_layout9 = QHBoxLayout()
        h_layout10 = QHBoxLayout()
        combined_layout = QHBoxLayout()
        combined_v_layout = QVBoxLayout()

        # We create the central widget and a vertical layout that will be used.
        # to place all the other widgets.
        # Central widget
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # We create a text field to display the path of the weight file and a button to select the file.
        # The button is bound to the select_file_weight method.

        # New text field to display file path.
        self.file_path_edit_weight = QLineEdit()
        self.file_path_edit_weight.setPlaceholderText("1.Select a weight file with an ending: .pt")
        self.file_path_edit_weight.setFixedHeight(50)
        self.file_path_edit_weight.setFixedWidth(650)
        h_layout2.addWidget(self.file_path_edit_weight)

        # File selection button.
        file_button_weight = QPushButton("File Search weight")
        file_button_weight.clicked.connect(self.select_file_weight)
        file_button_weight.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px;")
        file_button_weight.setFixedWidth(400)
        h_layout2.addWidget(file_button_weight)
        h_layout2.addStretch()

        # Similar to above, we create a text field and a button to select the dataset file.
        # The button is bound to the select_file_dataset method.
        # New text field to display file path.
        self.file_path_edit_dataset = QLineEdit()
        self.file_path_edit_dataset.setPlaceholderText("2. Select a dataset file with the extension: .yaml")
        self.file_path_edit_dataset.setFixedHeight(50) 
        self.file_path_edit_dataset.setFixedWidth(650)
        h_layout7.addWidget(self.file_path_edit_dataset)

        # File selection button.
        file_button_dataset = QPushButton("File Search dataset")
        file_button_dataset.clicked.connect(self.select_file_dataset)
        file_button_dataset.setStyleSheet("background-color: #2196F3; color: white; border-radius: 10px; padding: 10px;")
        file_button_dataset.setFixedWidth(400)
        h_layout7.addWidget(file_button_dataset)
        h_layout7.addStretch()

        # We add the layouts to the main vertical layout of the central widget.
        layout.addLayout(h_layout2)
        layout.addLayout(h_layout7)
        layout.addLayout(h_layout1)
        layout.addLayout(h_layout3)
        layout.addLayout(combined_layout)
        combined_layout.addLayout(combined_v_layout)
        layout.addLayout(h_layout8)
        layout.addLayout(h_layout10)
        layout.addLayout(h_layout6)
        layout.addLayout(h_layout9)


        # We create a text field for the input of the source (camera).
        # We define the minimum height and width of the field and add it to the first horizontal layout.
        label_input = QLabel("3. Enter input source (camera)")
        label_input.setFixedWidth(650)
        self.combo_box_input = QComboBox()
        self.combo_box_input.setFixedHeight(50)
        self.combo_box_input.setFixedWidth(200)
        self.combo_box_input.addItem("0")
        self.combo_box_input.addItem("1")
        self.combo_box_input.addItem("2")
        h_layout1.addWidget(label_input)
        h_layout1.addWidget(self.combo_box_input)
        h_layout1.addStretch()

        # We create additional text fields for entering the device, recognition score and maximum
        # number of items. We add these fields to the corresponding horizontal layouts.
        label_device = QLabel("4. Select Device: 0,1,2,3 for GPU or cpu")
        label_device.setFixedWidth(650)
        self.combo_box_device = QComboBox()
        self.combo_box_device.setFixedHeight(50)
        self.combo_box_device.setFixedWidth(200)
        self.combo_box_device.addItem("0")
        self.combo_box_device.addItem("1")
        self.combo_box_device.addItem("2")
        self.combo_box_device.addItem("3")
        self.combo_box_device.addItem("cpu")
        h_layout3.addWidget(label_device)
        h_layout3.addWidget(self.combo_box_device)
        h_layout3.addStretch()

        # Add photo
        self.image_label = QLabel()
        self.image_label.setPixmap(QPixmap("images2.png"))
        self.image_label.setScaledContents(True)
        self.image_label.setFixedWidth(450)
        self.image_label.setFixedHeight(250)
        self.image_label.setStyleSheet("padding: 5px;")
        combined_layout.addWidget(self.image_label)
        combined_layout.addStretch()

        # Replaced QLineEdit with QSlider and QLabel
        label_slider = QLabel("5. Select Recognition SCORE")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(99)
        self.slider.setValue(10)
        self.slider.setTickInterval(5)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setSingleStep(5)
        self.slider.valueChanged.connect(self.update_label)
        self.slider.setFixedWidth(650)
        self.label = QLabel("0.10")
        self.label.setMinimumHeight(50)
        self.label.setFixedWidth(500)
        combined_v_layout.addWidget(label_slider)
        combined_v_layout.addWidget(self.slider)
        combined_v_layout.addWidget(self.label)

        self.text_edit2 = QLineEdit()
        self.text_edit2.setPlaceholderText("6. Enter the maximum number of objects to recognize here")
        self.text_edit2.setFixedHeight(50)
        self.text_edit2.setFixedWidth(650)
        self.text_edit2.setAlignment(Qt.AlignLeft)
        combined_v_layout.addWidget(self.text_edit2)

        # We create a selection list (ComboBox) and add recognizable object options.
        # We bind the selection change with the combo_box_changed method.
        # Select list class
        self.combo_box = QComboBox()
        self.combo_box.setMaximumHeight(50)
        self.combo_box.setMaximumWidth(300)

        # Add options to the picklist
        options = [
            "Man", "Bicycle", "Car", "Motorcycle", "Plane", "Bus", "Train", "Truck", "Boat", "Traffic Light",
		"Fire Horn", "STOP Sign", "Parking Meter", "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow",
 		"Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Bag", "Tie", "Suitcase", "Frisbee", "Ski",
 		"Snowboard", "Sports Ball", "Kite", "Baseball Bat", "Baseball Glove", "Skateboard", "Surfboard",
 		"Tennis Racket", "Bottle", "Wine Glass", "Mug", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
 		"Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Chair", "Couch",
 		"Potted plant", "Bed", "Dining table", "Toilet", "TV", "Laptop", "Mouse", "Remote control", "Keyboard",
 		"Mobile Phone", "Microwave Oven", "Oven", "Toaster", "Sink", "Fridge", "Book", "Watch", "Flower Pot",
 		"Scissors", "Teddy Bear", "Hair Dryer", "Toothbrush"
        ]
        self.combo_box.addItem(f'7. Select Class', -1)
        for i, option in enumerate(options):
            self.combo_box.addItem(option, i)
        # Apply styling to QComboBox
        self.combo_box.setStyleSheet("""
            QComboBox {
                background-color: #black;
                color: #007bff;
                border: 1px solid #007bff;
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border-left-width: 1px;
                border-left-color: #000000;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox QAbstractItemView {
                background-color: #light-grey;
                color: #000000;
                selection-background-color: #007bff;
                selection-color: #ffffff;
            }
        """)
        self.combo_box.currentIndexChanged.connect(self.combo_box_changed)
        h_layout8.addWidget(self.combo_box)
        spacer = QSpacerItem(150, 0, QSizePolicy.Fixed, QSizePolicy.Minimum)
        h_layout8.addItem(spacer)

        # We create a checkbox to disable video storage.
        # We bind the state change to the checkbox_toggled method.
        self.checkbox = QCheckBox("* Disable video storage")
        self.checkbox.stateChanged.connect(self.checkbox_toggled)
        h_layout8.addWidget(self.checkbox)
        h_layout8.addStretch()

        # We create a button to store the values ​​entered in the text fields.
        # We bind the button with the save_text and show_message methods.
        # Save source button
        save_button = QPushButton("8. Saving values")
        save_button.clicked.connect(self.save_text)
        save_button.clicked.connect(self.show_message)
        save_button.setStyleSheet("background-color: #9C27B0; color: white; border-radius: 10px; padding: 10px;")
        save_button.setFixedWidth(300)
        h_layout10.addWidget(save_button)
        spacer = QSpacerItem(10, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_layout10.addItem(spacer)

        # We create two buttons to start object recognition and tracking with camera movement.
        # We bind the buttons to the methods the_button_was_clicked και the_button1_was_clicked.
        # Activation buttons
        button = QPushButton("START RECOGNITION")
        button.setCheckable(True)
        button.clicked.connect(self.the_button_was_clicked)
        button.setStyleSheet("background-color: #FFC107; color: black; border-radius: 10px; padding: 10px;")
        button.setFixedWidth(400)
        h_layout6.addWidget(button)

        button1 = QPushButton("START RECOGNITION WITH CAMERA MOTION")
        button1.setCheckable(True)
        button1.clicked.connect(self.the_button1_was_clicked)
        button1.setStyleSheet("background-color: #f44336; color: white; border-radius: 10px; padding: 10px;")
        button1.setFixedWidth(700)
        h_layout6.addWidget(button1)
        h_layout6.addStretch()

        # We create an exit button that closes the application when pressed.
        # Add the termination icon.
        left_spacer = QSpacerItem(200, 0, QSizePolicy.Fixed, QSizePolicy.Minimum)
        h_layout9.addItem(left_spacer)
        # Create QPushButton with arrow image
        self.exit_button = QPushButton()
        icon = QIcon("arrow_down1.png") 
        self.exit_button.setIcon(icon)
        scaling_factor = 0.15 
        pixmap = QPixmap("arrow_down1.png")
        scaled_pixmap = pixmap.scaled(int(pixmap.width() * scaling_factor), int(pixmap.height() * scaling_factor), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.exit_button.setFixedSize(scaled_pixmap.size())
        self.exit_button.setIconSize(scaled_pixmap.size())
        self.exit_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                padding: 0;
            }
        """)
        self.exit_button.clicked.connect(self.close)
        h_layout9.addWidget(self.exit_button)
        right_spacer = QSpacerItem(10, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_layout9.addItem(right_spacer)

        # We create a ScrollArea to contain the main widget,
        # allowing scrolling if the content exceeds the window size.
        # Adding the layout to a QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidget(central_widget)
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)

    # They handle storing the values, running the detect.py and detect_pan.py scripts,
    # selecting files and displaying messages.

    def combo_box_changed(self, index):
        global x
        # x = index
        x = self.combo_box.itemData(index)
        if x == -1:
            print("Επιλέξτε μια κλάση")
        else:
            print(f'Επιλέχθηκε: {x}')

    # Search File Weight
    def select_file_weight(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;Python Files (*.py)", options=options)
        if file_path:
            self.file_path_edit_weight.setText(file_path)
            global text_weight
            text_weight = file_path
            print("File path saved:", file_path)

    # Browse file DataSet
    def select_file_dataset(self):
        options = QFileDialog.Options()
        file_path1, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;Python Files (*.py)", options=options)
        if file_path1:
            self.file_path_edit_dataset.setText(file_path1)
            global text_dataset
            text_dataset = file_path1
            print("File path saved:", file_path1)

    # Button: Saving values
    def show_message(self):
        msg = QMessageBox()
        msg.setWindowTitle("Saving values")
        msg.setText(f'The following were saved.. \nFor the threshold: {text_weight} \nFor the input source: {text} \nFor the Computer device: {y} \nFor the recognition SCORE: {thress} \nFor the maximum number of objects to identify: {max_det}')
        msg.exec_()

    # Message: Item save message popup
    def save_text(self):
        global text
        global y
        global thress
        global max_det
        text = self.combo_box_input.currentText()
        y = self.combo_box_device.currentText()
        thress = self.label.text()
        max_det = self.text_edit2.text()
        print(f'The following were saved.. \nFor the threshold: {text_weight} \nFor the input source: {text} \nFor the Computer device {y} \nFor the recognition SCORE: {thress} \nFor the maximum number of objects to identify: {max_det}')


    # CheckBox: Disable video storage
    def checkbox_toggled(self):
        global save
        if self.checkbox.isChecked():
            save = "--nosave"
        else:
            save = ""

    def update_label(self, value):
        self.label.setText(f"{value / 100:.2f}")

    # Button: Start object recognition
    def the_button_was_clicked(self):
        global x, text, thress, max_det, y, text_weight
        if text_weight == 0:
            QMessageBox.warning(self, "Error", "Please select a file weight!!!")
        elif x == -1:
            QMessageBox.warning(self, "Error", "Please select a class!!!")
        elif not max_det:
            QMessageBox.warning(self, "Error", "Please fill in the maximum number of items!!")
        else:
            os.system(f'python3 detect.py --source {text} --weights  {text_weight}  --data {text_dataset} --classes {x} --conf-thres {thress}  --max-det {max_det} --device {y} {save}')
            print(f'python3 detect.py --source {text} --weights  {text_weight}  --data {text_dataset} --classes {x} --conf-thres {thress}  --max-det {max_det} --device {y} {save}')

    # Button: Start object recognition and tracking with camera movement.
    def the_button1_was_clicked(self):
        global x, text, thress, max_det, y, text_weight
        if text_weight == 0:
            QMessageBox.warning(self, "Error", "Please select a file weight!!!")
        elif x == -1:
            QMessageBox.warning(self, "Error", "Please select a class!!!")
        else:
            os.system(f'python3 detect_pan.py --source {text} --weights {text_weight}  --data {text_dataset} --classes {x} --conf-thres {thress} --max-det {max_det} --device {y}')
            print(f'python3 detect_pan.py --source {text} --weights {text_weight}  --data {text_dataset} --classes {x} --conf-thres {thress} --max-det {max_det} --device {y}')

# We create the application, create and display the main window and start
# the main application execution loop.
app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec_())
