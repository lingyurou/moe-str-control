import torch
from PIL import Image
import torchvision.transforms as T
from models import GMoESTRModel
from data.tokenizer import CTCTokenizer


def predict(img_path, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CTCTokenizer()

    # 初始化模型并载入权重
    model = GMoESTRModel().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    # 处理单张图像
    transform = T.Compose([
        T.Resize((32, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 补充 Batch 维度 [1, 3, 32, 128]

    with torch.no_grad():
        logits, _ = model(image)  # logits: [T, 1, num_classes]
        preds_index = logits.argmax(2).permute(1, 0)  # [1, T]
        pred_str = tokenizer.decode(preds_index)

    print(f"🔮 模型预测识别结果: '{pred_str[0]}'")


if __name__ == "__main__":
    # 示例调用
    # predict("data/test.png", "weights/checkpoints/g_moe_str_epoch_49.pth")
    pass