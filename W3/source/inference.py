import source.global_variables as gv
from ultralytics import YOLO


def post_process_results(results):
    processed_results = []
    for i, result in enumerate(results):
        bbox_list = []
        for bb in result.boxes.data:
            # Return in format [bb_left, bb_top, bb_width, bb_height, confidence]
            bbox_list.append([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]), float(bb[4])])
        processed_results.append(bbox_list)
    return processed_results


def predict():
    yolo_model = YOLO(gv.PATH_TO_MODEL)
    results = yolo_model.predict(
        source=gv.PATH_TO_VIDEO,
        stream=True,
    )
    results = post_process_results(results)
    return results
