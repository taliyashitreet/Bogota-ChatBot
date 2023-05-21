# Text Classification Project

## Project Overview
This project is centered around building a deep learning model for text classification. We aimed to classify text into six distinct categories:

1. Land use
2. Other
3. Environment and climate resilience
4. Local identity
5. Future of work
6. Mobility (transport)

We leveraged the power of neural networks for this task and trained our model using the TensorFlow framework. The model was trained to comprehend the context of the text and classify it into the appropriate category.

## Model Training and Evaluation
Our model was compiled using the binary cross-entropy loss function and was evaluated during training using various metrics including:

- Binary Accuracy
- Precision
- Recall
- F1 Score

These metrics helped us understand how well our model was performing for each category independently. The model was trained over 16 epochs.

Post training, the model was evaluated on a separate test set. We computed the accuracy for each category independently. The results are as follows:

- Category 1 (Land use): Accuracy of 85.19%
- Category 2 (Other): Accuracy of 88.89%
- Category 3 (Environment and climate resilience): Accuracy of 81.48%
- Category 4 (Local identity): Accuracy of 85.19%
- Category 5 (Future of work): Accuracy of 92.59%
- Category 6 (Mobility (transport)): Accuracy of 96.30%

These results indicate that our model performed quite well in classifying the text into different categories, with accuracy for individual categories ranging from 81.48% to 96.30%.

## Conclusion
This project exhibits the effectiveness of deep learning in text classification tasks. The model shows promising performance in classifying text into six categories. Future work can include tuning the model for better performance, adding more categories, or applying the model to different text classification tasks.