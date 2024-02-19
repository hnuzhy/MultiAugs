# MultiAugs
Code for my paper "Boosting Semi-Supervised 2D Human Pose Estimation by Revisiting Data Augmentation and Consistency Training"

## Abstract 
*The 2D human pose estimation is a basic visual problem. However, supervised learning of a model requires massive labeled images, which is expensive and labor-intensive. In this paper, we aim at boosting the accuracy of a pose estimator by excavating extra unlabeled images in a semi-supervised learning (SSL) way. Most previous consistency-based SSL methods strive to constraint the model to predict consistent results for differently augmented images. Following this consensus, we revisit two core aspects including **advanced data augmentation methods** and **concise consistency training frameworks**. Specifically, we heuristically dig various **collaborative combinations** of existing data augmentations, and discover novel superior data augmentation schemes to more effectively add noise on unlabeled samples. They can compose easy-hard augmentation pairs with larger transformation difficulty gaps, which play a crucial role in consistency-based SSL. Moreover, we propose to **strongly augment unlabeled images repeatedly with diverse augmentations**, generate multi-path predictions sequentially, and optimize corresponding unsupervised consistency losses using one single network. This simple and compact design is on a par with previous methods consisting of dual or triple networks. Undoubtedly, it can also be integrated with multiple networks to produce better performance. Comparing to state-of-the-art SSL approaches, our method brings substantial improvements on public datasets.*

## Brief Description




## 
