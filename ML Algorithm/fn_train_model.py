import torch
import time


def train_model(model, train, test, loss_fn, optim, epochs, bs, update_freq):
    train_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    total_train = train.shape[0]
    total_test = test.shape[0]
    train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test, batch_size=test.shape[0], shuffle=False)

    main_tic = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tic = time.perf_counter()
        loss_train = 0
        correct_train = 0
        for imgs in train_loader:
            labels = imgs[:, -1].long()
            imgs = imgs[:, :-1]

            batch_size = imgs.shape[0]
            outputs = model(imgs.view(batch_size, -1))
            _, predicted = torch.max(outputs, dim=1)
            correct_train += int((predicted == labels).sum())
            loss = loss_fn(outputs, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_train += loss.item()

        toc = time.perf_counter()

        correct_val = 0
        with torch.no_grad():
            for imgs in val_loader:
                labels = imgs[:, -1].long()
                imgs = imgs[:, :-1]
                batch_size = imgs.shape[0]
                outputs = model(imgs.view(batch_size, -1))
                _, predicted = torch.max(outputs, dim=1)
                correct_val += int((predicted == labels).sum())

        if epoch == 1 or epoch == epochs or epoch % update_freq == 0:
            print(
                f"Epoch {epoch}:\n\tDuration = {round(toc - tic, 3)} seconds\n\tTraining Loss: {round(loss_train / len(train_loader), 5)}\n\tTraining Accuracy: {round(correct_train / total_train, 3)}\n\tValidation Accuracy: {round(correct_val / total_test, 3)}")

        train_loss_hist.append(round(loss_train / len(train_loader), 5))
        train_acc_hist.append(round(correct_train / total_train, 5))
        test_acc_hist.append(round(correct_val / total_test, 5))

    main_toc = time.perf_counter()
    print(
        f"\nTotal Training Time = {round(main_toc - main_tic, 3)} seconds\nAverage Training Time per Epoch = {round((main_toc - main_tic) / epochs, 3)} seconds")
    return train_loss_hist, train_acc_hist, test_acc_hist
