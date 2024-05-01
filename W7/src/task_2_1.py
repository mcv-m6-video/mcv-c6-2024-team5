import argparse
import numpy as np

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Task 2.1')
    parser.add_argument('--predictions_1', type=str, default='results/x3Ds_best/predictions.npy')
    parser.add_argument('--predictions_2', type=str, default='results/x3Ds_best_OF/predictions.npy')
    parser.add_argument('--gt_labels', type=str, default='results/gt_labels.npy')
    parser.add_argument('--w1', type=float, default=0.4, help='Weight for predictions_1. Default: 0.5')
    parser.add_argument('--w2', type=float, default=0.6, help='Weight for predictions_2. Default: 0.5')
    args = parser.parse_args()

    # Load the predictions and ground truth labels
    predictions_1 = np.load(args.predictions_1)
    predictions_2 = np.load(args.predictions_2)

    final_predictions = args.w1 * predictions_1 + args.w2 * predictions_2

    gt_labels = np.load(args.gt_labels)

    # Get the argmax of all of the predictions
    predictions_1 = np.argmax(predictions_1, axis=1)
    predictions_2 = np.argmax(predictions_2, axis=1)
    final_predictions = np.argmax(final_predictions, axis=1)

    # Calculate the accuracy of the predictions
    accuracy_1 = np.mean(predictions_1 == gt_labels)
    accuracy_2 = np.mean(predictions_2 == gt_labels)
    accuracy_final = np.mean(final_predictions == gt_labels)

    # Print the accuracies
    print(f'Accuracy of predictions_1: {accuracy_1}')
    print(f'Accuracy of predictions_2: {accuracy_2}')
    print(f'Accuracy of final predictions: {accuracy_final}')
