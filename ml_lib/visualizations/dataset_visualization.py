import matplotlib.pyplot as plt


def show_dataset_metadata(train_dataset):
    print("total items on dataset: ",len(train_dataset))
    print("total classes on dataset: ",len(train_dataset.classes))
    rows,columns = (4,4)

    _,axs = plt.subplots(rows,columns)

    j =0
    for i in range(rows*columns):
        image,label = train_dataset[i]
        axs[i%rows,j].imshow(image, cmap="gray")
        axs[i%rows,j].set_title(f"Label: {label} ({train_dataset.classes[label]})")
        axs[i%rows,j].axis("off")
        if (i+1) % rows ==0:
            j= (j+1) % columns

    plt.show()
