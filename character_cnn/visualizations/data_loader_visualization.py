import matplotlib.pyplot as plt


def visualize_data_loader_batch(loader,dataset_classes):
    images, labels = next(iter(loader)) # calls next and takes the 'next' value on the loader (batch)
    print("image Shape :", images.shape)
    print("labels Shape:", labels.shape)

    _, axes = plt.subplots(2, 8, figsize=(12, 4))

    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze()  #<- removes the channel of dimension 1 (after images[i] result is 1x28x28) squeeze reduces to 28x28 the actual image
        img = img * 0.5 + 0.5  # remove normalization (inverse operation)

        ax.imshow(img, cmap="gray")
        ax.set_title(dataset_classes[labels[i].item()]) #<- item retrieves the 1 dimentional tenso value
        ax.axis("off")

    plt.tight_layout()
    plt.show()
