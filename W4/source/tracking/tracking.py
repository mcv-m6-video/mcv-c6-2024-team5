import numpy as np
import time as t
import torch
from .sort import Sort
from torchvision.transforms import transforms
from PIL import Image
import cv2


class Car:
    def __init__(self, x, y, w, h, confidence, embedding=None, last_update=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.embedding = embedding
        self.n_measures = 1
        self.last_update = last_update  # We will measure time in frames
        self.new = True

    def update(self, x, y, w, h, confidence, embedding, f_idx):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.n_measures += 1
        self.last_update = f_idx
        # We update the embedding by averaging all previous embeddings plus the new one
        if embedding is not None:
            self.embedding = (self.embedding * (self.n_measures - 1) + embedding) / self.n_measures


class Tracking:
    def __init__(self, iou_threshold=0.3, embedding_model=None, distance_threshold=20, c_distance_threshold=100):
        self.tracker = Sort(iou_threshold=iou_threshold)
        self.cars = {}
        self.embedding_model = embedding_model
        self.distance_threshold = distance_threshold
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                lambda x: x.float() / 255,
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.frame_threshold = 10
        self.c_distance_threshold = c_distance_threshold

    def get_distances(self, embedding):
        distances = []
        for car in self.cars.values():
            distances.append(np.power((car.embedding - embedding), 2).sum())
        return np.array(distances)

    def get_distances_to_previous(self, embedding, f_idx):
        distances = {id: np.power((car.embedding - embedding), 2).sum() for id, car in self.cars.items() if car.last_update < f_idx - 1}
        return distances

    def __call__(self, preds, frame, f_idx=0):
        np_pred = np.array(preds)
        if len(np_pred) == 0:
            np_pred = np.empty((0, 5))
        pred_with_ids = self.tracker.update(np_pred)
        # Transform them to list of lists
        pred_with_ids = pred_with_ids.astype(int).tolist()
        # Insert the confidence back in the 5th position of the list (index 4)
        for pred_with_id, p in zip(pred_with_ids, preds):
            aux = pred_with_id[4]
            pred_with_id[4] = p[4]
            pred_with_id.append(aux)

        if len(pred_with_ids) == 0:
            return
        # for car in pred_with_ids:
        #     cv2.imshow(f'Car {car[5]}', frame[car[1]:car[3], car[0]:car[2]])
        #     cv2.waitKey(0)
        bbox_images = torch.stack(
            [self.transform(Image.fromarray(frame[car[1]:car[3], car[0]:car[2]])) for car in pred_with_ids])
        embeddings = self.embedding_model(bbox_images).cpu().detach().numpy()
        for car, embedding in zip(pred_with_ids, embeddings):
            car_id = car[5]
            if car_id not in self.cars:
                self.cars[car_id] = Car(car[0], car[1], car[2], car[3], car[4], embedding, f_idx)
            else:
                self.cars[car_id].update(car[0], car[1], car[2], car[3], car[4], embedding, f_idx)

        to_remove = [car_id for car_id, car in self.cars.items() if f_idx - car.last_update > self.frame_threshold]
        for car_id in to_remove:
            del self.cars[car_id]

        # # First, we get the images of the bboxes
        # bbox_images = torch.stack([self.transform(Image.fromarray(frame[car[1]:car[3], car[0]:car[2]])) for car in pred_with_ids])
        # for car in pred_with_ids:
        #     car_id = car[5]
        #     # I don't even know what I am doing here, but it works and I am not going to touch it
        #     # I am sorry for the future me that will have to read this
        #     # With a combination of the embedding and the distance between centroids, it is decided if the car is a new car or not
        #     if car_id not in self.cars:
        #         embedding = self.embedding_model(bbox_images[0].unsqueeze(0)).cpu().detach().numpy()
        #         # First, we check if it is close to any other car in the dictionary
        #         distances = self.get_distances_to_previous(embedding, f_idx)
        #         centroid_distances = {id: np.sqrt((car[0] - c.x) ** 2 + (car[1] - c.y) ** 2) for id, c in self.cars.items()}
        #         if len(distances) == 0 or np.min(np.array(list(distances.values()))) >= self.distance_threshold:
        #             # We add the car to the dictionary
        #             self.cars[car_id] = Car(car[0], car[1], car[2], car[3], car[4], embedding, f_idx)
        #         else:
        #             # Find minimum distance to centroid
        #             min_distance = np.min(np.array(list(centroid_distances.values())))
        #             key_min = min(centroid_distances, key=centroid_distances.get)
        #             if min_distance < self.c_distance_threshold:
        #                 if key_min in distances:
        #                     new_car_id_exists = self.tracker.change_id(car_id, key_min)
        #                     if new_car_id_exists:
        #                         # We add the car to the dictionary
        #                         self.cars[car_id] = Car(car[0], car[1], car[2], car[3], car[4], embedding, f_idx)
        #                     else:
        #                         self.cars[key_min].update(car[0], car[1], car[2], car[3], car[4], embedding, f_idx)
        #                         car[5] = key_min
        #                 else:
        #                     self.cars[car_id] = Car(car[0], car[1], car[2], car[3], car[4], embedding, f_idx)
        #             else:
        #                 # We add the car to the dictionary
        #                 self.cars[car_id] = Car(car[0], car[1], car[2], car[3], car[4], embedding, f_idx)
        #     else:
        #         # We update the car
        #         self.cars[car_id].update(car[0], car[1], car[2], car[3], car[4], None, f_idx)
        #
        # # We remove the cars that have not been updated in the last self.frame_threshold frames
        # to_remove = [car_id for car_id, car in self.cars.items() if f_idx - car.last_update > self.frame_threshold]
        # for car_id in to_remove:
        #     del self.cars[car_id]
        # return pred_with_ids


class MultiCameraTracking:
    def __init__(self, tracking_dict):
        self.tracking_dict = tracking_dict

    def __call__(self, preds, frames, f_idx):
        for result, (id, frame) in zip(preds, frames.items()):
            self.tracking_dict[id](result, frame, f_idx)

        # We now have to check if the cars are the same in different cameras
        # For that, we will check the distance between embeddings of the cars in different cameras
        ordered_keys = sorted(self.tracking_dict.keys())
        for i in range(len(ordered_keys)):
            for j in range(i + 1, len(ordered_keys)):
                for car_id in self.tracking_dict[ordered_keys[i]].cars.keys():
                    distances = {}
                    car1 = self.tracking_dict[ordered_keys[i]].cars[car_id]
                    for car_id2 in self.tracking_dict[ordered_keys[j]].cars.keys():
                        car2 = self.tracking_dict[ordered_keys[j]].cars[car_id2]
                        distances[car_id2] = np.power((car1.embedding - car2.embedding), 2).sum()
                    if len(distances) > 0:
                        car_id2 = min(distances, key=distances.get)
                        if distances[car_id2] < 7:
                            car2 = self.tracking_dict[ordered_keys[j]].cars[car_id2]
                            if car2.new:
                                car2.new = False
                                # We get rid of the car in the second camera and add a new one with the same id in the first camera
                                self.tracking_dict[ordered_keys[i]].cars[car_id] = car1
                                car2 = self.tracking_dict[ordered_keys[j]].cars.pop(car_id2)
                                car2.last_update = f_idx
                                self.tracking_dict[ordered_keys[j]].cars[car_id] = car2
                                self.tracking_dict[ordered_keys[j]].tracker.change_id(car_id2, car_id)

        # Now, return al preds with ids
        new_preds = []
        for id, tracking in self.tracking_dict.items():
            preds_single_tracking = []
            for car_id, car in tracking.cars.items():
                if car.last_update == f_idx:
                    preds_single_tracking.append([car.x, car.y, car.w, car.h, car.confidence, car_id])
            new_preds.append(preds_single_tracking)
        return new_preds


