def calculate_accuracy(preds, gts):
    """计算学术界标准的 Word Accuracy (完全匹配准确率)"""
    correct = 0
    total = len(gts)
    for p, g in zip(preds, gts):
        if p.strip() == g.strip():
            correct += 1
    return correct / (total + 1e-6)