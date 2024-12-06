from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="yolo_config.yaml", epochs=1000, imgsz=640, save_period=10, patience = 20, cache=True,single_cls=True)

path = model.export(format="onnx")
