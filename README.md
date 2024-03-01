# HybridBranchNet

HybridBranchNet is a type of convolutional neural network (CNN)
architecture that was proposed in a research paper in 2023.
The main idea behind HybridBranchNet is to optimize
the design of CNNs in terms of depth, width,
and resolution based on a branch neural network


 The code for this article will soon be made available on the corresponding GitHub repository.

# HybridBranchNet Models Summary

## Overview
This research introduces the HybridBranchNet models, emphasizing the optimization of network architecture for balance between performance speed and accuracy. The models, HybridBranchNet1, HybridBranchNet2, and HybridBranchNet3, demonstrate incremental improvements by increasing the number of filters and branches. However, it was found that expanding beyond the HybridBranchNet3 model does not yield further optimization, suggesting a limit to scaling this architecture effectively.

## Key Findings
- **Model Architecture**: The number of filters in the HybridBranchNet models starts at 48, increasing by steps of 24. The HybridBranchNet series includes models with 3, 3, and 4 branches respectively.
- **Performance and Optimization**: Extending the architecture beyond the HybridBranchNet3 model does not contribute to further optimization, indicating an optimal point in the trade-off between performance speed and accuracy.
- **Comparison with Other Methods**: The proposed models outperform other methods in terms of the number of parameters and accuracy, as detailed in Table 5 and Figure 9. Specifically, HybridBranchNet models show superior accuracy and efficiency compared to EfficientNet and other conventional models across various metrics.

## Experimental Results
- The models were evaluated on a V100 system, ensuring consistent and reliable benchmarking.
- The inference time and training time were meticulously recorded, demonstrating the efficiency of HybridBranchNet models over others, including EfficientNet versions and traditional architectures like ResNet and DenseNet.

## Detailed Comparisons
- **EfficientNetV2 Comparison**: Table 6 highlights the comparison with the EfficientNetV2 series, showing that while EfficientNetV2 models have higher accuracy on the ImageNet dataset, HybridBranchNet models maintain a competitive edge with significantly fewer parameters and computational overhead.
- **Temporal Performance**: Figure 10 and Table 7 showcase the temporal efficiency of the HybridBranchNet models, indicating a substantial speed advantage over EfficientNet and other models.
- **State-of-the-Art Comparison**: The proposed method exhibits notable advantages in parameter efficiency and classification accuracy against state-of-the-art algorithms, as summarized in Table 8.

## Additional Insights
- The CIFAR-100 and Flowers-102 dataset comparisons further validate the effectiveness of the HybridBranchNet3 model, showing competitive accuracy with far fewer parameters.
- Preliminary experiments on ImageNet-1K with reduced training samples (Table 11) demonstrate the HybridBranchNet models' superior learning capability in scenarios with fewer data samples.

## Preprocessing Impact on Classification
The impact of preprocessing techniques on image classification was also explored, indicating significant improvements in image quality and classification accuracy. These techniques include contrast enhancement and edge improvement, informed by the works of KeGu and colleagues, as well as Mr. Chunwei Tian.

This summary encapsulates the core achievements and insights from the research on HybridBranchNet models, underlining their efficiency, effectiveness, and potential as a competitive architecture in deep learning for image classification.

## Detailed Results and Comparisons

### Accuracy Comparison on ImageNet

| Method              | Top-1 Acc (%) | Top-5 Acc (%) | #Params | Infer time (ms) | Train-time (hours) |
|---------------------|---------------|---------------|---------|-----------------|--------------------|
| HybridBranchNet0    | 78.3          | 94.1          | 5.1M    | 8               | 26                 |
| EfficientNetB0      | 77.1          | 93.3          | 5.3M    | 10              | 65                 |
| HybridBranchNet1    | 81.6          | 95.1          | 6.87M   | 13              | 30                 |
| EfficientNetB1      | 79.1          | 94.4          | 7.8M    | 23              | 93                 |
| HybridBranchNet2    | 82.7          | 95.8          | 8.11M   | 15              | 33                 |
| EfficientNetB2      | 80.1          | 94.9          | 9.2M    | 31              | 110                |
| HybridBranchNet3    | 83.1          | 96.7          | 9.12M   | 25              | 35                 |
| EfficientNetB3      | 81.6          | 95.8          | 12M     | 43              | 120                |
| ResNet              | 77.8          | 93.8          | 60M     | 56              | -                  |
| DenceNet            | 77.9          | 93.9          | 34M     | 52              | -                  |
| Inception           | 80.9          | 95.1          | 48M     | 60              | -                  |
| EfficientNetV2-S    | 83.9          | 97.3          | 22M     | 27              | 90                 |

### Temporal Performance Comparison

| Method            | model1 | model2 | model3 | model4 |
|-------------------|--------|--------|--------|--------|
| HybridBranchNet   | 8 ms   | 13 ms  | 15 ms  | 25 ms  |
| EfficientNet      | 10 ms  | 23 ms  | 31 ms  | 43 ms  |
| ResNet            | 22 ms  | 35 ms  | 42 ms  | 56 ms  |
| DenceNet          | 40 ms  | 52 ms  | NA     | NA     |
| Inception         | 45 ms  | 60 ms  | NA     | NA     |

### CIFAR-100 Dataset Comparison

| Model                     | Accuracy (%) | Parameters |
|---------------------------|--------------|------------|
| EffNet-L2                 | 96.08        | ≈ 480M     |
| MViT-B-16                 | 93.95        | ≈ 37M      |
| Oct-ResNet-152 (SE)       | 87.80        | ≈ 67M      |
| EfficientNet-b7 Tan and Le| 91.70        | ≈ 64M      |
| EfficientNetV2-L          | 92.26        | ≈ 120M     |
| EfficientNetV2-M          | 92.27        | ≈ 54M      |
| ResNet-50 (Fast AA)       | 83.64        | ≈ 25M      |
| DenseNet-264              | 83.95        | ≈ 105M     |
| HybridBranchNet3          | 92.30        | ≈ 9M       |

### Flowers-102 Dataset Comparison

| Model                     | Accuracy (%) | Parameters |
|---------------------------|--------------|------------|
| EffNet-L2                 | 99.65        | ≈ 480M     |
| MViT-B-16                 | 98.50        | ≈ 37M      |
| Oct-ResNet-152 (SE)       | 98.21        | ≈ 67M      |
| EfficientNet-b7 Tan and Le| 98.80        | ≈ 64M      |
| EfficientNetV2-L          | 98.80        | ≈ 120M     |
| EfficientNetV2-M          | 98.50        | ≈ 54M      |
| ResNet-50 (Fast AA)       | 97.90        | ≈ 25M      |
| HybridBranchNet3          | 98.80        | ≈ 9M       |

