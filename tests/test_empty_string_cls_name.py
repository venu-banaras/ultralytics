# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


"""Test script for the situation where class name strings are empty. This script tests the quality of life improvement of giving a user a warning about this issue, before they find it out while inferencing."""

from ultralytics import YOLO
from tests import SOURCE
import logging


def save_empty_cls_name_string_model(model, tmp_path):
    """Save a model with empty class names string.

    Args:
        model (YOLO): The YOLO model instance to modify and save.
        tmp_path (Path): Temporary directory path provided by pytest fixture.

    Returns:
        file: Saved model path
    """
    file = tmp_path / "model_empty_cls_name_string.pt"
    model.model.names = {0: "", 1: "cat", 2: "dog", 3: "elephant", 4: "", 5: "tiger", 6: "lion"}
    model.save(str(file))
    return str(file)


def test_empty_string_cls_name(isolated_model, tmp_path, caplog):
    """Test YOLO model where class name is an empty string and the user receives a warning for it.

    Args:
        isolated_model (Path): Path to isolated model fixture provided by pytest.
        tmp_path (Path): Temporary directory path provided by pytest fixture.
        caplog: Logging object provided by pytest fixture.
    """
    model = YOLO(isolated_model)

    # Save a YOLO model with empty class name string at indices 0 and 4
    empty_cls_name_string_model_path = save_empty_cls_name_string_model(model, tmp_path)
    empty_cls_name_string_model = YOLO(empty_cls_name_string_model_path)

    # Get the access to root looger of Ultralytics so as to capture the required warning message from console
    ultralytics_logger = logging.getLogger("ultralytics")
    ultralytics_logger.propagate = True

    # Use caplog to capture the warning
    with caplog.at_level(logging.WARNING, logger="ultralytics"):
        empty_cls_name_string_model(SOURCE, imgsz=32)
        assert "detected class ids" in caplog.text.lower()
