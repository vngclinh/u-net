import matplotlib.pyplot as plt
import torch
import os

def show_predictions(model, dataset, num=5, save=False, save_dir="results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(num):
            img, mask = dataset[i]  # img: [3, H, W], mask: [H, W]
            img_input = img.unsqueeze(0).to(next(model.parameters()).device)  # [1, 3, H, W]

            pred = model(img_input)                      # [1, C, H, W]
            pred_mask = torch.argmax(pred.squeeze(), dim=0).cpu().numpy()  # [H, W]

            # Prepare matplotlib figure
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(img.permute(1, 2, 0).cpu())     # [C, H, W] → [H, W, C]
            ax[0].set_title("Input Image")

            ax[1].imshow(mask.cpu(), cmap='gray')
            ax[1].set_title("True Mask")

            ax[2].imshow(pred_mask, cmap='gray')
            ax[2].set_title("Predicted Mask")

            for a in ax:
                a.axis("off")

            if save:
                save_path = os.path.join(save_dir, f"result_{i+1}.png")
                fig.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
                print(f"✅ Saved: {save_path}")
            else:
                plt.show()
                plt.close(fig)
