from fastai.vision.all import (
    URLs,
    untar_data,
    ImageDataLoaders,
    vision_learner,
    resnet18,
    accuracy,
)

if __name__ == "__main__":
    path = untar_data(URLs.CIFAR)
    dls = ImageDataLoaders.from_folder(path, train="train", valid="test")

    learn = vision_learner(dls, resnet18, metrics=accuracy)

    # Train the model
    learn.fit_one_cycle(1, lr_max=1e-3)

    learn.export(
        "data/model.fastai",
    )
