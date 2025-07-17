"""
Quantization Settings Dialog for GGUF Conversion
"""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QTextEdit,
    QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal
from .gguf_converter import GGUFConverterManager


class QuantizationDialog(QDialog):
    """Dialog for selecting GGUF conversion settings"""

    def __init__(self, model_path: str, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.converter_manager = GGUFConverterManager()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("GGUF Conversion Settings")
        self.setModal(True)
        self.setMinimumWidth(500)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Model info
        model_info_group = QGroupBox("Model Information")
        model_info_layout = QVBoxLayout()
        model_info_group.setLayout(model_info_layout)

        self.model_path_label = QLabel(f"Model: {self.model_path}")
        self.model_path_label.setWordWrap(True)
        model_info_layout.addWidget(self.model_path_label)

        # Validation
        is_valid, validation_msg = self.converter_manager.validate_model_for_conversion(
            self.model_path
        )
        self.validation_label = QLabel(
            f"Status: {'✅ Valid' if is_valid else '❌ Invalid'}"
        )
        self.validation_label.setStyleSheet(
            "color: green;" if is_valid else "color: red;"
        )
        model_info_layout.addWidget(self.validation_label)

        if not is_valid:
            self.validation_details = QLabel(validation_msg)
            self.validation_details.setStyleSheet("color: red; font-style: italic;")
            model_info_layout.addWidget(self.validation_details)

        layout.addWidget(model_info_group)

        # Quantization settings
        quantization_group = QGroupBox("Quantization Settings")
        quantization_layout = QVBoxLayout()
        quantization_group.setLayout(quantization_layout)

        # Quantization type
        quant_layout = QHBoxLayout()
        quant_layout.addWidget(QLabel("Quantization:"))
        self.quantization_combo = QComboBox()
        quantizations = self.converter_manager.get_quantization_options()
        for quant, description in quantizations.items():
            self.quantization_combo.addItem(f"{quant} - {description}", quant)
        self.quantization_combo.currentTextChanged.connect(self.update_size_estimate)
        quant_layout.addWidget(self.quantization_combo)
        quantization_layout.addLayout(quant_layout)

        # Context length
        context_layout = QHBoxLayout()
        context_layout.addWidget(QLabel("Context Length:"))
        self.context_length_spin = QSpinBox()
        self.context_length_spin.setRange(512, 32768)
        self.context_length_spin.setValue(4096)
        self.context_length_spin.setSingleStep(512)
        self.context_length_spin.valueChanged.connect(self.update_size_estimate)
        context_layout.addWidget(self.context_length_spin)
        quantization_layout.addLayout(context_layout)

        # Size estimation
        self.size_estimate_label = QLabel("Size estimate will be calculated...")
        self.size_estimate_label.setStyleSheet("color: #666; font-style: italic;")
        quantization_layout.addWidget(self.size_estimate_label)

        layout.addWidget(quantization_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.convert_btn = QPushButton("Start Conversion")
        self.convert_btn.clicked.connect(self.accept)
        self.convert_btn.setEnabled(is_valid)
        self.convert_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """
        )
        button_layout.addWidget(self.convert_btn)

        layout.addLayout(button_layout)

        # Calculate initial size estimate
        self.update_size_estimate()

    def update_size_estimate(self):
        """Update the size estimation display"""
        try:
            quantization = self.quantization_combo.currentData()
            input_mb, output_mb = self.converter_manager.estimate_conversion_size(
                self.model_path, quantization
            )

            if input_mb > 0 and output_mb > 0:
                compression_ratio = input_mb / output_mb
                self.size_estimate_label.setText(
                    f"Estimated: {input_mb:.1f}MB → {output_mb:.1f}MB "
                    f"(~{compression_ratio:.1f}x compression)"
                )
                self.size_estimate_label.setStyleSheet(
                    "color: #28a745; font-weight: bold;"
                )
            else:
                self.size_estimate_label.setText("Size estimate unavailable")
                self.size_estimate_label.setStyleSheet(
                    "color: #666; font-style: italic;"
                )
        except Exception as e:
            self.size_estimate_label.setText(f"Error calculating size: {str(e)}")
            self.size_estimate_label.setStyleSheet("color: red;")

    def get_conversion_settings(self) -> dict:
        """Get the selected conversion settings"""
        return {
            "quantization": self.quantization_combo.currentData(),
            "context_length": self.context_length_spin.value(),
        }
