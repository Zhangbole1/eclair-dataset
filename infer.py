from torchmetrics import JaccardIndex, F1Score, Accuracy

# 忽略标签 0
ignore_idx = 0
num_classes = cfg.num_classes

iou_metric = JaccardIndex(num_classes=num_classes, task="multiclass", ignore_index=ignore_idx)
f1_metric = F1Score(num_classes=num_classes, task="multiclass", average="none", ignore_index=ignore_idx)
acc_metric = Accuracy(num_classes=num_classes, task="multiclass", ignore_index=ignore_idx)

# 推理循环
model.eval()
for coords, features, labels in test_loader:
    coords, features, labels = coords.to(device), features.to(device), labels.to(device)
    with torch.no_grad():
        scores = model(coords, features)
        preds = scores.argmax(dim=1)
    iou_metric.update(preds, labels)
    f1_metric.update(preds, labels)
    acc_metric.update(preds, labels)

class_iou = iou_metric.compute()      # 各类 IoU
class_f1  = f1_metric.compute()       # 各类 F1
overall_acc = acc_metric.compute()    # 总体精度

print("Mean IoU:", class_iou[1:].mean())
print("Mean F1:", class_f1[1:].mean())
print("Overall Acc:", overall_acc)
